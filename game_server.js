/**
 * SpaceHuggers Headless Game Server
 *
 * Supports N parallel game instances in ONE process via vm.createContext().
 * One pipe round-trip serves all N envs — eliminates SubprocVecEnv overhead.
 *
 * Env vars:
 *   GAME_PATH   path to SpaceHuggers-main directory
 *   N_GAMES     number of parallel game instances (default 1)
 *
 * Protocol:
 *   Batch (N_GAMES > 1):
 *     → {"type":"step_batch","actions":[[m,v,sh,dg,gr]×N],"n":3}
 *     ← {"type":"states","states":[{...}×N]}
 *     → {"type":"reset_all"}
 *     ← {"type":"states","states":[{...}×N]}
 *     → {"type":"reset_batch","indices":[i,...]}
 *     ← {"type":"states","states":[{...}×len(indices)]}
 *
 *   Single (backward compat, targets instance 0):
 *     → {"type":"reset"}            ← {"type":"state","state":{...}}
 *     → {"type":"step","action":[...],"n":3}  ← {"type":"state","state":{...}}
 *
 *   Both modes:
 *     → {"type":"ping"}  ← {"type":"pong"}
 */

'use strict';

const fs       = require('fs');
const path     = require('path');
const vm       = require('vm');
const readline = require('readline');

const GAME_DIR = process.env.GAME_PATH || path.join(__dirname, 'SpaceHuggers-main');
const N_GAMES  = Math.max(1, parseInt(process.env.N_GAMES || '1'));

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Browser API factory helpers (one set per sandbox instance)
// ─────────────────────────────────────────────────────────────────────────────

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
        getBoundingClientRect() { return {left:0, right:w, top:0, bottom:h, width:w, height:h}; },
    };
    ctx2d.canvas = el;
    return el;
}

function makeDocument() {
    return {
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
}

function makeSandbox() {
    const sb = {
        _rafCallback: null,
        _frameCount:  0,
        innerWidth:   800,
        innerHeight:  600,
        chrome:       1,   // → lowGraphicsSettings=0, glOverlay=0 in app.js
        performance:  { now: () => Date.now() },
        location:     { href: '' },
        speechSynthesis:          { speak(){}, cancel(){} },
        SpeechSynthesisUtterance: class {},
        onkeydown: null, onkeyup: null, onmousedown: null, onmouseup: null,
        onmousemove: null, oncontextmenu: null, onwheel: null,
        ontouchstart: null, ontouchmove: null, ontouchend: null,
        navigator: { getGamepads(){ return []; }, userAgent: 'Node.js' },
        require, process, console,
    };
    sb.global = sb;  // 'global' inside scripts resolves to this sandbox
    sb.window = sb;  // 'window' too
    sb.requestAnimationFrame = (fn) => { sb._rafCallback = fn; };
    sb.document = makeDocument();
    sb.Image = class FakeImage {
        set src(_) {
            this.width = 128; this.height = 64;
            if (typeof this.onload === 'function') this.onload();
        }
    };
    sb.AudioContext = class FakeAudioContext {
        get currentTime()      { return sb._frameCount / 60; }
        createBuffer()         { return { getChannelData: () => new Float32Array(0) }; }
        createBufferSource()   { return { buffer:null, loop:false, connect(){}, start(){}, disconnect(){} }; }
        createGain()           { return { gain:{value:1}, connect(){} }; }
        get destination()      { return {}; }
        decodeAudioData(b, ok) { ok && ok({}); }
    };
    return sb;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Load & concatenate game source files (with patches)
// ─────────────────────────────────────────────────────────────────────────────

const FILES = [
    ['engine/engineUtil.js',      []],
    ['engine/engineDebug.js',     [['const debug = 1',       'const debug = 0'      ]]],
    ['engine/engine.js',          []],
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

let gameCode = '';
for (const [relPath, patches] of FILES) {
    let src = fs.readFileSync(path.join(GAME_DIR, relPath), 'utf8');
    for (const [from, to] of patches) src = src.replace(from, to);
    src = src.replace(/'use strict';\s*/g, '').replace(/"use strict";\s*/g, '');
    gameCode += `\n// ═══ ${relPath} ═══\n` + src;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Runtime helpers injected into every game context.
//     All game const/let/var are in scope (same concatenated script).
//     'global' resolves to the sandbox because makeSandbox() sets sb.global=sb.
//     Function declarations become properties of the sandbox → callable as
//     ctx._step(), ctx._setAIInput(), ctx._getGameState(), ctx.resetGame().
// ─────────────────────────────────────────────────────────────────────────────

const RUNTIME_FUNCS = `
function _step() {
    if (global._rafCallback) {
        var cb = global._rafCallback;
        global._rafCallback = null;
        global._frameCount++;
        cb(global._frameCount * (1000 / 60));
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
                var _sh = (Math.sign(dx) === facing && Math.abs(dx) < 8 && Math.abs(dy) < 1.2) ? 1 : 0;
                enemies.push({x:dx/lx, y:dy/ly, health:o.health/5, vx:(o.velocity?o.velocity.x:0)/0.2, vy:(o.velocity?o.velocity.y:0)/0.3, shootable:_sh, dist:dx*dx+dy*dy});
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
                var tile=getTileCollisionData(vec2(px+_dx, py+_dy)), val=0;
                if      (tile===tileType_ladder) val=-1.0;
                else if (tile===tileType_glass)  val= 0.5;
                else if (tile>0)                 val= 1.0;
                var _gi=(_dy+HH)*GW+(_dx+HW);
                grid[_gi]=val; gTerrain[_gi]=val;
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
        var bullets = [];
        for (var _bk=0; _bk<engineCollideObjects.length; _bk++) {
            var bb=engineCollideObjects[_bk];
            if (typeof Bullet!=='undefined' && bb instanceof Bullet && bb.team===team_enemy) {
                var bdx=bb.pos.x-p.pos.x, bdy=bb.pos.y-p.pos.y;
                bullets.push({x:bdx/lx, y:bdy/ly, vx:(bb.velocity?bb.velocity.x:0)/0.5, vy:(bb.velocity?bb.velocity.y:0)/0.5, dist:bdx*bdx+bdy*bdy});
            }
        }
        bullets.sort(function(a,b){ return a.dist-b.dist; });
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
        };
    } catch(e) { return null; }
}

// Wrap resetGame (defined as const/let in app.js, so NOT a sandbox property) in a
// function declaration so the outer scope can call ctx._resetAndStep() reliably.
function _resetAndStep() {
    resetGame();
    for (var _ri=0; _ri<5; _ri++) _step();
    return _getGameState();
}

// Bootstrap: a few frames so appUpdate/appRender have run at least once.
for (var _bi=0; _bi<5; _bi++) _step();
`;

const INSTANCE_CODE = gameCode + '\n' + RUNTIME_FUNCS;

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Create N isolated game instances
// ─────────────────────────────────────────────────────────────────────────────

const instances = [];
try {
    for (let i = 0; i < N_GAMES; i++) {
        const sandbox = makeSandbox();
        const ctx     = vm.createContext(sandbox);
        vm.runInContext(INSTANCE_CODE, ctx, { filename:`SpaceHuggers-${i}.js`, displayErrors:true });
        instances.push(ctx);
    }
} catch (err) {
    process.stderr.write('[game_server] FATAL during init: ' + err.stack + '\n');
    process.exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  stdin/stdout JSON message loop
// ─────────────────────────────────────────────────────────────────────────────

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on('line', function(rawLine) {
    var line = rawLine.trim();
    if (!line) return;
    var msg;
    try { msg = JSON.parse(line); } catch(e) { return; }

    if (msg.type === 'ping') {
        process.stdout.write('{"type":"pong"}\n');
        return;
    }

    // Batch: step all instances
    if (msg.type === 'step_batch') {
        var n = (msg.n && msg.n > 0) ? msg.n : 1;
        var states = instances.map(function(ctx, i) {
            var a = msg.actions[i];
            ctx._setAIInput(a[0], a[1], a[2], a[3], a[4]);
            for (var s = 0; s < n; s++) ctx._step();
            return ctx._getGameState();
        });
        process.stdout.write(JSON.stringify({type:'states', states:states}) + '\n');
        return;
    }

    // Batch: reset all instances
    if (msg.type === 'reset_all') {
        var states = instances.map(function(ctx) { return ctx._resetAndStep(); });
        process.stdout.write(JSON.stringify({type:'states', states:states}) + '\n');
        return;
    }

    // Batch: reset specific indices (episodes that ended)
    if (msg.type === 'reset_batch') {
        var states = msg.indices.map(function(i) { return instances[i]._resetAndStep(); });
        process.stdout.write(JSON.stringify({type:'states', states:states}) + '\n');
        return;
    }

    // Single-instance ops — backward compat with env_node.py (targets instances[0])
    if (msg.type === 'step') {
        var ctx = instances[0];
        var a = msg.action, n = (msg.n && msg.n > 0) ? msg.n : 1;
        ctx._setAIInput(a[0], a[1], a[2], a[3], a[4]);
        for (var s = 0; s < n; s++) ctx._step();
        process.stdout.write(JSON.stringify({type:'state', state:ctx._getGameState()}) + '\n');
        return;
    }

    if (msg.type === 'reset') {
        process.stdout.write(JSON.stringify({type:'state', state:instances[0]._resetAndStep()}) + '\n');
        return;
    }
});

rl.on('close', function() { process.exit(0); });
process.stderr.write('[game_server] ready\n');
