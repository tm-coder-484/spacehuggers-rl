/**
 * SpaceHuggers Headless Game Server — worker_threads edition
 *
 * Each game instance runs in its own OS thread (true V8 isolate parallelism).
 * The main thread coordinates N workers via MessageChannel pairs and handles
 * the stdin/stdout batch protocol.
 *
 * Env vars:
 *   GAME_PATH   path to SpaceHuggers-main directory
 *   N_GAMES     number of parallel game instances (default 1)
 *
 * Protocol (identical to game_server.js batch protocol):
 *   → {"type":"ping"}
 *   ← {"type":"pong"}
 *
 *   → {"type":"step_batch","actions":[[m,v,sh,dg,gr]×N],"n":3}
 *   ← {"type":"states","states":[{...}×N]}
 *
 *   → {"type":"reset_all"}
 *   ← {"type":"states","states":[{...}×N]}
 *
 *   → {"type":"reset_batch","indices":[i,...]}
 *   ← {"type":"states","states":[{...}×len(indices)]}
 */

'use strict';

const { Worker, isMainThread, workerData, MessageChannel } = require('worker_threads');
const fs       = require('fs');
const path     = require('path');
const vm       = require('vm');
const readline = require('readline');

// ─────────────────────────────────────────────────────────────────────────────
// Shared: game file list and patches (used in worker branch to build gameCode)
// ─────────────────────────────────────────────────────────────────────────────

const FILES = [
    ['engine/engineUtil.js',      []],
    ['engine/engineDebug.js',     [['const debug = 1',       'const debug = 0'      ]]],
    // Headless render-skip: skip the per-frame drawing (appRender + every
    // object's render() → canvas-2D calls through the noop Proxy, ~16ms/frame),
    // which RL never uses (it reads game STATE, not pixels).  IMPORTANT: the
    // engineObjects.sort() is KEPT because it reorders engineObjects and thus
    // the next engineUpdateObjects() iteration order — that must match the
    // original game exactly.  Gameplay randomness uses Math.random (rand()),
    // untouched by drawStars's randSeed manipulation, so skipping render does
    // not alter simulation behaviour.  Gated by _HEADLESS_NO_RENDER (RENDER=1
    // restores drawing for A/B measurement).
    ['engine/engine.js',          [[
          '        // render sort then render while removing destroyed objects\n'
        + '        glPreRender(mainCanvas.width, mainCanvas.height);\n'
        + '        appRender();\n'
        + '        engineObjects.sort((a,b)=> a.renderOrder - b.renderOrder);\n'
        + '        for(const o of engineObjects)\n'
        + '            o.destroyed || o.render();\n'
        + '        glCopyToContext(mainContext);\n'
        + '        appRenderPost();\n'
        + '        debugRender();',
          '        // [headless] keep the render-order sort (affects update order),\n'
        + '        // but skip all drawing — RL reads game state, not pixels.\n'
        + '        engineObjects.sort((a,b)=> a.renderOrder - b.renderOrder);\n'
        + '        if (!_HEADLESS_NO_RENDER) {\n'
        + '        glPreRender(mainCanvas.width, mainCanvas.height);\n'
        + '        appRender();\n'
        + '        for(const o of engineObjects)\n'
        + '            o.destroyed || o.render();\n'
        + '        glCopyToContext(mainContext);\n'
        + '        appRenderPost();\n'
        + '        debugRender();\n'
        + '        }'
    ]]],
    ['engine/engineObject.js',    []],
    ['engine/engineWebGL.js',     [['const glEnable = 1',    'const glEnable = 0'   ]]],
    ['engine/engineDraw.js',      []],
    ['engine/engineInput.js',     []],
    ['engine/engineAudio.js',     [['const soundEnable = 1', 'const soundEnable = 0']]],
    ['engine/engineTileLayer.js', []],
    ['engine/engineParticle.js',  []],
    ['appObjects.js',             []],
    ['appCharacters.js',          []],
    ['appEffects.js',             []],
    ['appLevel.js',               [['const warmUpTime = 2',  'const warmUpTime = 0' ]]],
    ['app.js',                    []],
];

// Runtime helpers injected after game code.
// Function declarations land on the worker's global via vm.runInThisContext().
// resetGame() is defined as const/let in app.js and lives in the concatenated
// script's lexical scope — _resetAndStep wraps it so it's accessible here.
const RUNTIME_FUNCS = `
// ── Headless particle stillbirth (perf) ─────────────────────────────────────
// Particles are pure cosmetics in headless: they are not observed, never
// collide with gameplay objects, and their only side effect
// (persistentParticleDestroyCallback) draws to the tile layer via the noop
// canvas proxy. BUT emitParticle() consumes a fixed sequence of rand() calls
// that gameplay code shares, so we cannot skip spawning without shifting the
// RNG stream. Instead: spawn (identical RNG draws), then mark destroyed
// immediately — engine.js filters destroyed objects at end of frame, so the
// particle never pays a single update(). Gate on _HEADLESS_NO_RENDER;
// RENDER=1 A/B runs keep full particle simulation.
// ── Headless tile-layer draw skip (perf) ────────────────────────────────────
// TileLayer.setData() stores gameplay-readable tile data AND redraws the tile
// to an offscreen canvas. The canvas half is pure pixels (noop Proxy headless)
// but still allocates vec2s + runs context calls per destroyed tile — the
// agent shoots constantly, so this is hot. Keep the data writes; skip draws.
if (typeof TileLayer !== 'undefined' && global._HEADLESS_NO_RENDER) {
    TileLayer.prototype.drawTileData    = function() {};
    TileLayer.prototype.drawAllTileData = function() {};
    TileLayer.prototype.redraw          = function() {};
    TileLayer.prototype.drawTile        = function() {};
    TileLayer.prototype.drawCanvas2D    = function() {};
}
if (typeof ParticleEmitter !== 'undefined' && global._HEADLESS_NO_RENDER) {
    const _origEmitParticle = ParticleEmitter.prototype.emitParticle;
    ParticleEmitter.prototype.emitParticle = function() {
        const p = _origEmitParticle.call(this);
        p.destroyCallback = 0;   // decal draw is a noop headless anyway
        p.destroyed = 1;         // removed by engine.js end-of-frame filter
        return p;
    };
}

function _step() {
    if (global._rafCallback) {
        var cb = global._rafCallback;
        global._rafCallback = null;
        global._frameCount++;
        cb(global._frameCount * (1000 / 60));
        _reapParticles();
    }
}

// Headless particle reaper (perf fix, ~3x+ sps).
// Particle.render() is the ONLY place particles set destroyed=1 (when
// age >= lifeTime). With rendering skipped (_HEADLESS_NO_RENDER), particles
// were immortal: engineObjects grew unboundedly (~4/frame) and every frame's
// update loop dragged thousands of zombie particles, so sim speed degraded
// linearly within an episode. Replicate just the expiry side-effect here.
// destroyCallback is preserved so e.g. persistent-decal spawns still fire,
// keeping headless behaviour identical to the rendered game.
function _reapParticles() {
    if (!global._HEADLESS_NO_RENDER) return;
    for (var i = engineObjects.length; i--;) {
        var o = engineObjects[i];
        if (o.lifeTime && !o.destroyed && o.spawnTime != undefined
            && time - o.spawnTime >= o.lifeTime) {
            o.destroyCallback && o.destroyCallback(o);
            o.destroyed = 1;
        }
    }
}

function _setAIInput(m, v, sh, dg, gr) {
    var left  = m === 1 ? 1 : 0;
    var right = m === 2 ? 1 : 0;
    var up    = v === 1 ? 1 : 0;
    var down  = v === 2 ? 1 : 0;
    function set(key, val) {
        var wasDown = inputData[0][key] ? inputData[0][key].d : 0;
        inputData[0][key] = {
            d: val ? 1 : 0,
            p: (val && !wasDown) ? 1 : 0,
            r: (!val && wasDown) ? 1 : 0,
        };
    }
    set(37, left);  set(39, right); set(38, up);  set(40, down);
    set(90, sh);    set(88, dg);    set(67, gr);
    isUsingGamepad = 0;
}

// Bounded BFS from (sx,sy) over passable tiles, in a (2R+1)^2 window centred on
// (cx,cy).  Returns a distance grid (dd, -1 = unreachable).  Used to build a
// geodesic "flow field" toward the nearest enemy — the agent's navigation signal.
// JUMP-HEIGHT model: the agent can ascend ~8 tiles through open air from a launch
// surface (player-tested max ~8 with a double-jump + good timing), or climb walls /
// ladders. So an UP move into open air is only valid if a solid/ladder launch
// surface lies within _JUMP_REACH below, or a climbable wall is adjacent. Without
// this the field promised impossible "float straight up" routes to high lone
// platforms (and straight into ceilings), and the agent got stuck under them.
var _JUMP_REACH = 8;
// Direct tile-collision read: identical semantics to getTileCollisionData
// (arrayCheck -> 0 out of bounds) but no Vector2 allocation. Nav hot path only.
function _tcGet(x, y) {
    var W = tileCollisionSize.x;
    return (x>=0 && y>=0 && x<W && y<tileCollisionSize.y) ? tileCollision[y*W+x] : 0;
}
function _canAscendTo(x, y) {
    if (_tcGet(x-1,y)>0 || _tcGet(x+1,y)>0) return true;  // wall-climb
    for (var k=1; k<=_JUMP_REACH; k++) {
        var t=_tcGet(x, y-k);
        if (t>0 || t===tileType_ladder) return true;   // solid ground / ladder to launch from
    }
    return false;
}

// DIG-AWARE weighted shortest-path (Dial's algorithm). Destructible terrain is
// PASSABLE-WITH-COST equal to the expected bullets-to-breach, mirroring the
// bullet destroy-chance in appObjects.collideWithTile (glass 100%, dirt 20%,
// base/pipe/solid 5%). This makes buried enemies reachable, so geoDist becomes a
// "dig-distance" and the existing geodesic reward rewards digging toward them.
var _bfsBufs = {};   // per-R reused buffers: dd (Int16) + asc memo (Int8)
function _bfsFrom(sx, sy, cx, cy, R) {
    var SZ=2*R+1, minx=cx-R, miny=cy-R, maxx=cx+R, maxy=cy+R;
    var buf = _bfsBufs[SZ] || (_bfsBufs[SZ] = { dd: new Int16Array(SZ*SZ), asc: new Int8Array(SZ*SZ) });
    var dd = buf.dd; dd.fill(-1);
    var asc = buf.asc; asc.fill(0);   // 0 unknown, 1 yes, 2 no
    function gi(x,y){ return (y-miny)*SZ+(x-minx); }
    // DIRECTIONAL edge cost = real traversal TIME. The gun fires HORIZONTALLY only,
    // so destructible tiles can only be dug on a move with a horizontal component; a
    // pure up/down move into rock/dirt/glass is impossible (can't shoot the ceiling
    // or floor) and is blocked. Empty/ladder is free in any direction (walk / climb /
    // jump / fall). Costs grounded in game physics: fireRate=8/s, walk~12 tiles/s.
    //   hard wall: 5% destroy/shot -> ~20 shots -> ~2.5s -> ~20-30 walk-tiles
    //   dirt:      20% -> ~5 shots  -> ~0.6s -> ~5 walk-tiles (+cascade, so a touch less)
    //   glass:     1 shot           -> ~0.1s -> ~1 walk-tile
    // Walk cost stays 1 (== old binary-BFS unit step) so walkable enemies produce an
    // IDENTICAL signal to the pre-dig model -> perfect warm-start transfer.
    function cost(fx, fy, tx, ty){
        var t=_tcGet(tx,ty);
        if (t<=0 || t===tileType_ladder) return 1;   // empty / ladder: walk, climb, jump, fall
        if (tx===fx) return -1;                       // pure vertical into a destructible tile: can't dig up/down
        if (t===tileType_glass) return 1;             // shoot through window (horizontal)
        if (t===tileType_dirt) return 5;              // dig dirt (~5 shots)
        return 20;                                     // dig base/pipe/solid (~20 shots)
    }
    var base={dd:dd,SZ:SZ,minx:minx,miny:miny,maxx:maxx,maxy:maxy,gi:gi,ok:false};
    if (sx<minx||sx>maxx||sy<miny||sy>maxy) return base;
    var MAXC=96, buckets={};   // > GEO_NORM so geo_dir still points toward reachable-but-far buried enemies (whose geo_dist caps at 1.0)
    var s0=gi(sx,sy); dd[s0]=0; buckets[0]=[s0];
    for (var dcur=0; dcur<=MAXC; dcur++) {
        var bk=buckets[dcur]; if (!bk) continue;
        for (var bi=0; bi<bk.length; bi++) {
            var idx=bk[bi];
            if (dd[idx]!==dcur) continue;               // stale (already relaxed lower)
            var x=minx+(idx%SZ), y=miny+((idx/SZ)|0);
            for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
                if (!ax && !ay) continue;
                var nx=x+ax, ny=y+ay;
                if (nx<minx||nx>maxx||ny<miny||ny>maxy) continue;
                var nidx=gi(nx,ny);
                var c=cost(x,y,nx,ny);
                if (c<0) continue;                      // blocked (e.g. pure-vertical dig)
                if (ny>y && _tcGet(nx,ny)===tileType_empty) {   // memoised jump-reach check
                    var av=asc[nidx];
                    if (!av) { av = _canAscendTo(nx,ny) ? 1 : 2; asc[nidx]=av; }
                    if (av===2) continue;                       // can't jump that high through open air
                }
                var nd=dcur+c;
                if (nd<=MAXC && (dd[nidx]===-1 || nd<dd[nidx])) {
                    dd[nidx]=nd;
                    (buckets[nd]||(buckets[nd]=[])).push(nidx);
                }
            }
        }
    }
    base.ok=true; return base;
}

// ── COARSE global flow-field ─────────────────────────────────────────────────
// A low-res BFS over the WHOLE level so the agent can be routed the long way
// around obstacles the local R=24 field can't see (e.g. an enemy above a solid
// ceiling reachable only via a 100-tile detour). The coarse passability map is
// built ONCE per level (terrain is static apart from digging) and cached; the
// BFS runs each frame on the small coarse grid. Cells of all-solid rock become
// blocked, so routes go AROUND rock masses. Used only when the fine field fails.
var _COARSE_CELL = 4, _COARSE_NORM = 120, _COARSE_MAXC = 240;
function _buildCoarseMap() {
    var lx = levelSize ? Math.ceil(levelSize.x) : 200;
    var ly = levelSize ? Math.ceil(levelSize.y) : 100;
    var CW = Math.max(1, Math.ceil(lx/_COARSE_CELL)), CH = Math.max(1, Math.ceil(ly/_COARSE_CELL));
    var cost = new Array(CW*CH);
    for (var cy=0; cy<CH; cy++) for (var cx=0; cx<CW; cx++) {
        var open=0, dirt=0;
        for (var ty=0; ty<_COARSE_CELL; ty++) for (var tx=0; tx<_COARSE_CELL; tx++) {
            var t=_tcGet(cx*_COARSE_CELL+tx, cy*_COARSE_CELL+ty);
            if (t<=0 || t===tileType_ladder || t===tileType_glass) open++;
            else if (t===tileType_dirt) dirt++;
        }
        // cost to ENTER: any open space -> cheap walk; dirt-only -> diggable; else blocked
        cost[cy*CW+cx] = open>0 ? 1 : (dirt>0 ? 4 : -1);
    }
    return {cost:cost, CW:CW, CH:CH};
}
function _coarseField(px, py, ex, ey) {
    var res = {dist:1.0, dx:0, dy:0, ok:false};
    if (typeof levelSize === 'undefined' || !levelSize) return res;
    if (global._coarseLevel !== level) { global._coarseMap = _buildCoarseMap(); global._coarseLevel = level; }
    var cm = global._coarseMap, CW = cm.CW, CH = cm.CH, cost = cm.cost;
    var sx=Math.floor(px/_COARSE_CELL), sy=Math.floor(py/_COARSE_CELL);
    var tx=Math.floor(ex/_COARSE_CELL), ty=Math.floor(ey/_COARSE_CELL);
    if (sx<0||sx>=CW||sy<0||sy>=CH||tx<0||tx>=CW||ty<0||ty>=CH) return res;
    var N=CW*CH, dd=new Array(N); for (var i=0;i<N;i++) dd[i]=-1;
    var buckets={}; dd[sy*CW+sx]=0; buckets[0]=[sy*CW+sx];
    for (var dcur=0; dcur<=_COARSE_MAXC; dcur++) {
        var bk=buckets[dcur]; if (!bk) continue;
        for (var bi=0; bi<bk.length; bi++) {
            var idx=bk[bi]; if (dd[idx]!==dcur) continue;
            var x=idx%CW, y=(idx/CW)|0;
            for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
                if (!ax && !ay) continue;
                var nx=x+ax, ny=y+ay;
                if (nx<0||nx>=CW||ny<0||ny>=CH) continue;
                var c=cost[ny*CW+nx]; if (c<0) continue;          // blocked (solid rock)
                var nidx=ny*CW+nx, nd=dcur+c;
                if (nd<=_COARSE_MAXC && (dd[nidx]===-1 || nd<dd[nidx])) { dd[nidx]=nd; (buckets[nd]||(buckets[nd]=[])).push(nidx); }
            }
        }
    }
    var ed=dd[ty*CW+tx];
    if (ed<0) return res;                                          // enemy cell unreachable even coarsely
    res.ok=true; res.dist=Math.min(1, ed/_COARSE_NORM);
    var bx=tx, by=ty, guard=4000;                                  // steepest-descent backtrack -> first macro step
    while (guard-->0) {
        var cd=dd[by*CW+bx]; if (cd<=0) break;
        var best=cd, bnx=bx, bny=by;
        for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
            if (!ax && !ay) continue;
            var nx=bx+ax, ny=by+ay;
            if (nx<0||nx>=CW||ny<0||ny>=CH) continue;
            var v=dd[ny*CW+nx];
            if (v>=0 && v<best) { best=v; bnx=nx; bny=ny; }
        }
        if (bnx===bx && bny===by) break;
        if (best===0) break;
        bx=bnx; by=bny;
    }
    res.dx=Math.sign(bx-sx); res.dy=Math.sign(by-sy);
    return res;
}

function _getGameState() {
    try {
        var p = players && players[0];
        if (!p) return null;
        var lx = levelSize ? levelSize.x : 200;
        var ly = levelSize ? levelSize.y : 100;
        var facing = p.mirror ? -1 : 1;   // gun fires horizontally in this direction
        var enemies = [];
        for (var _ei = 0; _ei < engineCollideObjects.length; _ei++) {
            var o = engineCollideObjects[_ei];
            if (o.isCharacter && o.team === team_enemy && !o.isDead()) {
                var dx = o.pos.x - p.pos.x;
                var dy = o.pos.y - p.pos.y;
                // "shootable": enemy is in the facing direction, within bullet
                // range (~8 tiles) and vertically aligned (bullets are horizontal).
                var _sh = (Math.sign(dx) === facing && Math.abs(dx) < 8 && Math.abs(dy) < 1.2) ? 1 : 0;
                var _aim = (Math.sign(-dx) === (o.mirror?-1:1)) ? (o.holdingShoot?1.0:0.4) : 0.0;
                enemies.push({x:dx/lx, y:dy/ly, health:o.health/5, vx:(o.velocity?o.velocity.x:0)/0.2, vy:(o.velocity?o.velocity.y:0)/0.3, shootable:_sh, aiming:_aim, dist:dx*dx+dy*dy});
            }
        }
        enemies.sort(function(a,b){ return a.dist-b.dist; });
        var GW=13, GH=9, HW=6, HH=4;
        var px=Math.round(p.pos.x), py=Math.round(p.pos.y);
        var grid     = new Array(GW*GH).fill(0);   // v1: terrain + objects (kept)
        var gTerrain = new Array(GW*GH).fill(0);   // v2 channel: terrain
        var gEnemy   = new Array(GW*GH).fill(0);   // v2 channel: enemy occupancy
        var gHazard  = new Array(GW*GH).fill(0);   // v2 channel: hazards + bullets
        for (var _dy=-HH; _dy<=HH; _dy++) {
            for (var _dx=-HW; _dx<=HW; _dx++) {
                var tile=getTileCollisionData(vec2(px+_dx, py+_dy));
                var val=0, dval=0;                    // val: v1 grid (binary-ish); dval: v2 dig-cost
                if      (tile===tileType_ladder) { val=-1.0; dval=-1.0; }
                else if (tile===tileType_glass)  { val= 0.5; dval= 0.25; }  // window: 1 shot to breach
                else if (tile===tileType_dirt)   { val= 1.0; dval= 0.5;  }  // dirt: ~5 shots (+cascade)
                else if (tile>0)                 { val= 1.0; dval= 1.0;  }  // base/pipe/solid: ~20 shots
                var _gi=(_dy+HH)*GW+(_dx+HW);
                grid[_gi]=val; gTerrain[_gi]=dval;
            }
        }
        for (var _oi=0; _oi<engineCollideObjects.length; _oi++) {
            var oo=engineCollideObjects[_oi];
            if (oo.isGameObject && !oo.isCharacter && !oo.isWeapon && !oo.isCheckpoint && !oo.destroyed) {
                var odx=Math.round(oo.pos.x-px), ody=Math.round(oo.pos.y-py);
                if (Math.abs(odx)<=HW && Math.abs(ody)<=HH) {
                    grid[(ody+HH)*GW+(odx+HW)] = oo.explosionSize>0 ? 0.75 : 0.3;
                    gHazard[(ody+HH)*GW+(odx+HW)] = oo.explosionSize>0 ? 1.0 : 0.5;
                }
            }
        }
        for (var _qi=0; _qi<engineCollideObjects.length; _qi++) {   // enemy occupancy
            var qe=engineCollideObjects[_qi];
            if (qe.isCharacter && qe.team===team_enemy && !qe.isDead()) {
                var qdx=Math.round(qe.pos.x-px), qdy=Math.round(qe.pos.y-py);
                if (Math.abs(qdx)<=HW && Math.abs(qdy)<=HH)
                    gEnemy[(qdy+HH)*GW+(qdx+HW)] = Math.max(0.2, qe.health/5);
            }
        }
        for (var _rk=0; _rk<engineCollideObjects.length; _rk++) {   // enemy bullets -> hazard
            var rb=engineCollideObjects[_rk];
            if (typeof Bullet!=='undefined' && rb instanceof Bullet && rb.team===team_enemy) {
                var rdx=Math.round(rb.pos.x-px), rdy=Math.round(rb.pos.y-py);
                if (Math.abs(rdx)<=HW && Math.abs(rdy)<=HH)
                    gHazard[(rdy+HH)*GW+(rdx+HW)] = 1.0;
            }
        }
        // nearest incoming ENEMY bullets (pos + velocity) — for reactive dodging
        var bullets = [];
        for (var _bk=0; _bk<engineCollideObjects.length; _bk++) {
            var bb=engineCollideObjects[_bk];
            if (typeof Bullet!=='undefined' && bb instanceof Bullet && bb.team===team_enemy) {
                var bdx=bb.pos.x-p.pos.x, bdy=bb.pos.y-p.pos.y;
                bullets.push({x:bdx/lx, y:bdy/ly, vx:(bb.velocity?bb.velocity.x:0)/0.5, vy:(bb.velocity?bb.velocity.y:0)/0.5, dist:bdx*bdx+bdy*bdy});
            }
        }
        bullets.sort(function(a,b){ return a.dist-b.dist; });
        // ── geodesic flow field toward the nearest enemy (navigation) ─────────
        // One BFS FROM the player over a bounded region. Gives: a local
        // reachability channel (dist-from-player), the geodesic distance to the
        // nearest enemy, and the first-step direction toward it (backtracked).
        var R=24, GEO_NORM=48, REACH_NORM=16;   // exact old binary-BFS scale -> walkable-enemy signals are byte-identical on warm-start
        var gReach = new Array(GW*GH).fill(1.0);
        var geoDist=1.0, geoDx=0, geoDy=0;
        var fld = global._WANT_GEO ? _bfsFrom(px, py, px, py, R) : {ok:false};
        if (fld.ok) {
            if (enemies.length) {
                var ne=enemies[0];
                var etx=Math.round(p.pos.x + ne.x*lx), ety=Math.round(p.pos.y + ne.y*ly);
                if (etx>=fld.minx&&etx<=fld.maxx&&ety>=fld.miny&&ety<=fld.maxy) {
                    var ed=fld.dd[fld.gi(etx,ety)];
                    if (ed>=0) {
                        geoDist=Math.min(1, ed/GEO_NORM);
                        // steepest-descent backtrack along the weighted cheapest path:
                        // step to the neighbour with the smallest dist until we reach
                        // the player's neighbour, giving the first dig/move direction.
                        var bx=etx, by=ety, guard=4000;
                        while (guard-->0) {
                            var cd=fld.dd[fld.gi(bx,by)];
                            if (cd<=0) break;
                            var best=cd, bnx=bx, bny=by;
                            for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
                                if (!ax&&!ay) continue;
                                var nx=bx+ax, ny=by+ay;
                                if (nx<fld.minx||nx>fld.maxx||ny<fld.miny||ny>fld.maxy) continue;
                                var v=fld.dd[fld.gi(nx,ny)];
                                if (v>=0 && v<best) { best=v; bnx=nx; bny=ny; }
                            }
                            if (bnx===bx && bny===by) break;   // no descent available
                            if (best===0) break;               // neighbour is player -> (bx,by) is first step
                            bx=bnx; by=bny;
                        }
                        geoDx=Math.sign(bx-px); geoDy=Math.sign(by-py);
                    }
                }
            }
            for (var wy=-HH;wy<=HH;wy++) for (var wx=-HW;wx<=HW;wx++) {
                var tx=px+wx, ty=py+wy, val=1.0;
                if (tx>=fld.minx&&tx<=fld.maxx&&ty>=fld.miny&&ty<=fld.maxy) {
                    var dv=fld.dd[fld.gi(tx,ty)];
                    if (dv>=0) val=Math.min(1, dv/REACH_NORM);
                }
                gReach[(wy+HH)*GW+(wx+HW)]=val;
            }
        }
        // Long-range routing: if the fine geodesic field couldn't reach the nearest
        // enemy (beyond R), use the COARSE whole-level field to route the long way
        // around obstacles. coarse_dist (detour-aware) feeds the long-range reward.
        // Straight-line is only a last resort if even the coarse field finds no path.
        var coarseDist = 1.0;
        if (geoDx===0 && geoDy===0 && enemies.length) {
            var _ne0 = enemies[0];
            var _cf = _coarseField(px, py, p.pos.x + _ne0.x*lx, p.pos.y + _ne0.y*ly);
            if (_cf.ok && (_cf.dx!==0 || _cf.dy!==0)) {
                geoDx = _cf.dx; geoDy = _cf.dy; coarseDist = _cf.dist;
            } else {
                geoDx = Math.sign(_ne0.x); geoDy = Math.sign(_ne0.y);
            }
        }
        // ── coarse egocentric GLOBAL map (see beyond the local window) ────────
        // 8x8 cells, 8 tiles each (±32 tiles). 3 channels: terrain, enemies-now,
        // decaying enemy memory (remembers where enemies were, resets per level).
        var GG=8, CELL=8, half=GG/2;
        var gTerr=new Array(GG*GG).fill(0), gNow=new Array(GG*GG).fill(0);
        if (global._WANT_GEO) {
            for (var cyi=0;cyi<GG;cyi++) for (var cxi=0;cxi<GG;cxi++) {
                var sx=Math.round(p.pos.x + (cxi-half+0.5)*CELL);
                var sy=Math.round(p.pos.y + (cyi-half+0.5)*CELL);
                var tt=getTileCollisionData(vec2(sx,sy));
                gTerr[cyi*GG+cxi] = (tt>0 && tt!==tileType_ladder) ? 1 : 0;
            }
            for (var _gei=0;_gei<engineCollideObjects.length;_gei++) {
                var ge=engineCollideObjects[_gei];
                if (ge.isCharacter && ge.team===team_enemy && !ge.isDead()) {
                    var gcx=Math.floor((ge.pos.x-p.pos.x)/CELL)+half;
                    var gcy=Math.floor((ge.pos.y-p.pos.y)/CELL)+half;
                    if (gcx>=0&&gcx<GG&&gcy>=0&&gcy<GG)
                        gNow[gcy*GG+gcx]=Math.min(1, gNow[gcy*GG+gcx]+0.5);
                }
            }
        }
        if (global._gmemLevel!==level) { global._gmem=new Array(GG*GG).fill(0); global._gmemLevel=level; }
        var gm=global._gmem;
        for (var _mi=0;_mi<GG*GG;_mi++) gm[_mi]=Math.min(1, gm[_mi]*0.97 + gNow[_mi]);
        var globe = gTerr.concat(gNow).concat(gm.slice());   // 192 = 3*8*8

        return {
            px:p.pos.x/lx, py:p.pos.y/ly,
            vx:p.velocity.x/0.2, vy:p.velocity.y/0.3,
            health:p.health, ground:p.groundTimer.active()?1.0:0.0,
            grenades:(p.grenadeCount||0)/3.0,
            dodge_ready:!p.dodgeRechargeTimer.active()?1.0:0.0,
            on_fire:p.burnTimer.active()?1.0:0.0,
            alive:!p.isDead(), lives:playerLives, kills:totalKills,
            level:level, warmup:levelWarmup?1:0,
            enemies:enemies.slice(0,10).map(function(e){return [e.x,e.y,e.health];}),
            enemy_vels:enemies.slice(0,10).map(function(e){return [e.vx,e.vy];}),
            enemies_remaining:(typeof levelEnemyCount!=='undefined'?Math.max(0,levelEnemyCount):0)+enemies.length,
            facing:facing,
            enemy_shootable:enemies.slice(0,10).map(function(e){return e.shootable;}),
            bullets:bullets.slice(0,5).map(function(b){return [b.x,b.y,b.vx,b.vy];}),
            n_enemies:enemies.length, grid:grid,
            grid_terrain:gTerrain, grid_enemy:gEnemy, grid_hazard:gHazard,
            grid_reach:gReach, geo_dist:geoDist, geo_dir:[geoDx, geoDy],
            coarse_dist:coarseDist,
            enemy_aiming:enemies.slice(0,10).map(function(e){return e.aiming;}),
            globe:globe,
        };
    } catch(e) { return null; }
}

// _resetAndStep wraps resetGame (a const/let in app.js, not a global property)
// so it is accessible via closure within this same concatenated script.
function _resetAndStep() {
    global._coarseLevel = null;   // force coarse-map rebuild: a new level-1 layout reuses level==1
    resetGame();
    for (var _ri=0; _ri<5; _ri++) _step();
    return _getGameState();
}

// Bootstrap: a few frames so appUpdate/appRender have run at least once.
for (var _bi=0; _bi<5; _bi++) _step();
`;

// ─────────────────────────────────────────────────────────────────────────────
// WORKER BRANCH
// ─────────────────────────────────────────────────────────────────────────────

if (!isMainThread) {
    const { port, gameDir, workerIndex } = workerData;

    // Set up browser-API mocks on this worker's global.
    // Node.js 21+ defines 'navigator' and 'performance' as read-only getters
    // on the global object (matching browser behaviour), so plain assignment
    // throws.  Use Object.defineProperty to override; fall back to assignment
    // for any property that is already freely writable.
    function defGlobal(name, value) {
        try {
            Object.defineProperty(global, name, {
                value, writable: true, configurable: true, enumerable: true,
            });
        } catch (_) {
            try { global[name] = value; } catch (_2) {}
        }
    }

    // Headless render-skip flag (see engine.js patch in FILES). Default ON;
    // set RENDER=1 in the environment to keep the render path for A/B testing.
    defGlobal('_HEADLESS_NO_RENDER', process.env.RENDER === '1' ? 0 : 1);
    // Geodesic flow-field BFS only runs when requested (v2 obs). Saves the BFS
    // cost for v1/flat runs on this backend.
    defGlobal('_WANT_GEO', process.env.GEO === '1' ? 1 : 0);

    defGlobal('navigator',   { getGamepads() { return []; }, userAgent: 'Node.js' });
    defGlobal('performance', { now: () => Date.now() });
    defGlobal('location',    { href: '' });
    defGlobal('window',      global);
    defGlobal('global',      global);
    defGlobal('chrome',      1);   // → lowGraphicsSettings=0, glOverlay=0 in app.js
    defGlobal('_rafCallback', null);
    defGlobal('_frameCount',  0);
    defGlobal('innerWidth',   800);
    defGlobal('innerHeight',  600);
    defGlobal('speechSynthesis',           { speak() {}, cancel() {} });
    defGlobal('SpeechSynthesisUtterance',  class {});
    defGlobal('onkeydown',    null); defGlobal('onkeyup',       null);
    defGlobal('onmousedown',  null); defGlobal('onmouseup',     null);
    defGlobal('onmousemove',  null); defGlobal('oncontextmenu', null);
    defGlobal('onwheel',      null);
    defGlobal('ontouchstart', null); defGlobal('ontouchmove',   null);
    defGlobal('ontouchend',   null);
    defGlobal('requestAnimationFrame', (fn) => { global._rafCallback = fn; });

    // Canvas / document mocks
    function makeDeepNoop() {
        const store = {};
        const proxy = new Proxy(store, {
            get(t, k) {
                if (k in t) return t[k];
                if (typeof k === 'symbol') return undefined;
                return (..._args) => proxy;
            },
            set(t, k, v) { t[k] = v; return true; },
        });
        return proxy;
    }

    function makeCanvas(w, h) {
        w = w || 800; h = h || 600;
        const ctx2d = makeDeepNoop();
        const el = {
            width: w, height: h, style: {},
            getContext(type) { return type === '2d' ? ctx2d : null; },
            getBoundingClientRect() { return { left:0, right:w, top:0, bottom:h, width:w, height:h }; },
        };
        ctx2d.canvas = el;
        return el;
    }

    global.document = {
        createElement(tag) {
            if (tag === 'canvas') return makeCanvas();
            return { style:{}, appendChild(){}, tagName:tag.toUpperCase(),
                     innerHTML:'', disabled:false, value:'', type:'',
                     oninput:null, onclick:null };
        },
        createTextNode() { return {}; },
        body: { appendChild(){}, style:'' },
        hasFocus() { return true; },
    };

    global.Image = class FakeImage {
        set src(_) {
            this.width = 128; this.height = 64;
            if (typeof this.onload === 'function') this.onload();
        }
    };

    global.AudioContext = class FakeAudioContext {
        get currentTime()      { return global._frameCount / 60; }
        createBuffer()         { return { getChannelData: () => new Float32Array(0) }; }
        createBufferSource()   { return { buffer:null, loop:false, connect(){}, start(){}, disconnect(){} }; }
        createGain()           { return { gain:{ value:1 }, connect(){} }; }
        get destination()      { return {}; }
        decodeAudioData(b, ok) { ok && ok({}); }
    };

    // Build and run game code in this worker's global context.
    let gameCode = '';
    try {
        for (const [relPath, patches] of FILES) {
            let src = fs.readFileSync(path.join(gameDir, relPath), 'utf8');
            for (const [from, to] of patches) src = src.replace(from, to);
            src = src.replace(/'use strict';\s*/g, '').replace(/"use strict";\s*/g, '');
            gameCode += `\n// ═══ ${relPath} ═══\n` + src;
        }
    } catch (err) {
        process.stderr.write(`[worker ${workerIndex}] FATAL reading game files: ${err.stack}\n`);
        process.exit(1);
    }

    try {
        vm.runInThisContext(gameCode + '\n' + RUNTIME_FUNCS, {
            filename: `SpaceHuggers-worker-${workerIndex}.js`,
            displayErrors: true,
        });
    } catch (err) {
        process.stderr.write(`[worker ${workerIndex}] FATAL during init: ${err.stack}\n`);
        process.exit(1);
    }

    // Signal readiness to the main thread.
    port.postMessage({ type: 'ready' });

    // Handle step/reset messages from the main thread.
    port.on('message', (msg) => {
        try {
            if (msg.type === 'step') {
                _setAIInput(msg.action[0], msg.action[1], msg.action[2], msg.action[3], msg.action[4]);
                for (let s = 0; s < msg.n; s++) _step();
                port.postMessage({ type: 'state', json: JSON.stringify(_getGameState()) });
            } else if (msg.type === 'reset') {
                port.postMessage({ type: 'state', json: JSON.stringify(_resetAndStep()) });
            }
        } catch (err) {
            process.stderr.write(`[worker ${workerIndex}] error handling message: ${err.stack}\n`);
            port.postMessage({ type: 'state', json: 'null' });
        }
    });

    // Keep the worker alive (port keeps the event loop running).
    return;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN THREAD BRANCH
// ─────────────────────────────────────────────────────────────────────────────

const GAME_DIR = process.env.GAME_PATH || path.join(__dirname, 'SpaceHuggers-main');
const N_GAMES  = Math.max(1, parseInt(process.env.N_GAMES || '1', 10));

// Spawn N workers, each with its own MessageChannel port pair.
// workerData carries the worker-side port plus config.
const ports = [];   // main-thread ends of the MessageChannel pairs

const workers = [];
for (let i = 0; i < N_GAMES; i++) {
    const { port1, port2 } = new MessageChannel();
    ports.push(port1);

    const worker = new Worker(__filename, {
        workerData: {
            port:        port2,
            gameDir:     GAME_DIR,
            workerIndex: i,
        },
        // Transfer port2 into the worker so it can be used there.
        transferList: [port2],
    });

    worker.on('error', (err) => {
        process.stderr.write(`[worker ${i}] uncaught error: ${err.stack}\n`);
    });
    worker.on('exit', (code) => {
        if (code !== 0) {
            process.stderr.write(`[worker ${i}] exited with code ${code}\n`);
        }
    });

    workers.push(worker);
}

// Wait for all N workers to be ready before accepting protocol messages.
async function waitForReady() {
    await Promise.all(
        ports.map((port, i) =>
            new Promise((resolve) => {
                port.once('message', (msg) => {
                    if (msg.type === 'ready') resolve();
                    else {
                        process.stderr.write(`[worker ${i}] unexpected first message: ${JSON.stringify(msg)}\n`);
                        resolve();
                    }
                });
            })
        )
    );
}

// Send a message to all workers and collect responses in parallel.
function sendAll(msgs) {
    return Promise.all(
        ports.map((port, i) =>
            new Promise((resolve) => {
                port.once('message', (resp) => resolve(resp.json));
                port.postMessage(msgs[i]);
            })
        )
    );
}

// Send a message to a specific subset of workers and collect their responses.
function sendSubset(indices, msgs) {
    return Promise.all(
        indices.map((i, j) =>
            new Promise((resolve) => {
                ports[i].once('message', (resp) => resolve(resp.json));
                ports[i].postMessage(msgs[j]);
            })
        )
    );
}

async function runServer() {
    await waitForReady();
    process.stderr.write('[game_server_workers] ready\n');

    const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

    // for-await ensures one message is fully processed before the next begins,
    // preventing any possibility of interleaved concurrent handlers.
    for await (const rawLine of rl) {
        const line = rawLine.trim();
        if (!line) continue;

        let msg;
        try { msg = JSON.parse(line); } catch (e) { continue; }

        if (msg.type === 'ping') {
            process.stdout.write('{"type":"pong"}\n');
            continue;
        }

        if (msg.type === 'step_batch') {
            const n = (msg.n && msg.n > 0) ? msg.n : 1;
            const msgs = ports.map((_, i) => ({
                type:   'step',
                action: msg.actions[i],
                n,
            }));
            const states = await sendAll(msgs);
            process.stdout.write('{"type":"states","states":[' + states.join(',') + ']}\n');
            continue;
        }

        if (msg.type === 'reset_all') {
            const msgs = ports.map(() => ({ type: 'reset' }));
            const states = await sendAll(msgs);
            process.stdout.write('{"type":"states","states":[' + states.join(',') + ']}\n');
            continue;
        }

        if (msg.type === 'reset_batch') {
            const indices = msg.indices;
            const msgs    = indices.map(() => ({ type: 'reset' }));
            const states  = await sendSubset(indices, msgs);
            process.stdout.write('{"type":"states","states":[' + states.join(',') + ']}\n');
            continue;
        }
    }

    // stdin closed — shut down workers cleanly
    for (const w of workers) w.terminate();
    process.exit(0);
}

runServer().catch((err) => {
    process.stderr.write('[game_server_workers] fatal: ' + err.stack + '\n');
    process.exit(1);
});
