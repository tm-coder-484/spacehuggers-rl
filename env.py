"""
SpaceHuggers gymnasium environment — wraps the game via Playwright.

Reward design philosophy (v3 — clean, no penalty traps):
  Primary signals  : kills (+10), level completion (+150), death (-5)
  Dense shaping    : approach reward ±0.05/step to guide toward enemies
  Mechanic bonuses : fire-dodge (+0.1), ladder use (+0.003/0.015), wall-climb (+0.01)
  Survival         : removed (was +0.002/step — paid for passive turtling on long episodes)

  NO action penalties — exploration must remain free.  Penalising actions
  (shoot, grenade, position) was causing entropy collapse: the agent learned
  "doing nothing is safest" and stopped exploring entirely.

Observations (141 floats):
  player_x, player_y, vx, vy, health, on_ground,
  grenade_count (0-3), dodge_ready (0/1), on_fire (0/1)  (9)
  nearest 5 enemies: rel_x, rel_y, health               (15)
  13x9 tile grid centered on player                     (117)
    -1.0 = ladder   0.0 = empty    0.3 = inert prop
     0.5 = glass   0.75 = explosive  1.0 = solid

Actions  MultiDiscrete([3, 3, 2, 2, 2]):
  [0] horizontal : 0=none  1=left        2=right
  [1] vertical   : 0=none  1=up/jump     2=down (ladder)
  [2] shoot      : 0=no    1=yes
  [3] dodge      : 0=no    1=yes   (X — recharges in ~2 s, puts out fire)
  [4] grenade    : 0=no    1=throw (C — refills to 3 on each respawn)

SpaceHuggers quick-ref:
  Enemy health : Recruit=1, Soldier=2, Captain=3, Specialist=4, DemoExpert=5
  Player lives : start 10, +3 per level completed
  Gun range    : ~10-15 tiles (get in close!)
  Roll in air  : gives upward speed boost (= double jump)
  Roll melee   : rolling into an enemy does damage + brief invulnerability
"""

import time
import numpy as np
import gymnasium as gym
from pathlib import Path
from playwright.sync_api import sync_playwright

# v2 (Tier-2) Dict obs builder + dims, reused from the node-workers env so the
# browser-rendered watch path produces identical observations to training.
from env_node_batch import (_to_obs_v2, GRID_W as V2_GRID_W, GRID_H as V2_GRID_H,
                            GRID_CH_V2, GLOBE_CH, GLOBE_SZ, VEC_DIM_V2)

_INJECT = """
(function() {
    // Prevent engine from wiping inputData when window loses focus
    document.hasFocus = () => true;

    window._setAIInput = function(left, right, up, down, shoot, dodge, grenade) {
        const set = (key, val) => {
            const wasDown = inputData[0][key]?.d || 0;
            inputData[0][key] = {
                d: val ? 1 : 0,
                p: val && !wasDown ? 1 : 0,
                r: !val && wasDown ? 1 : 0,
            };
        };
        set(37, left);      // left arrow
        set(39, right);     // right arrow
        set(38, up);        // up arrow  — jump + climb up ladders
        set(40, down);      // down arrow — climb down ladders
        set(90, shoot);     // Z = shoot
        set(88, dodge);     // X = dodge roll
        set(67, grenade);   // C = throw grenade
        isUsingGamepad = 0;
    };

    // ── jump-height model (mirrors game_server_workers.js) ────────────────────
    // The agent can ascend ~8 tiles through open air from a launch surface, or
    // climb walls/ladders. An UP move into open air is only valid if a solid/ladder
    // launch surface lies within _JUMP_REACH below, or a climbable wall is adjacent.
    window._JUMP_REACH = 8;
    window._canAscendTo = function(x, y) {
        if (getTileCollisionData(vec2(x-1,y))>0 || getTileCollisionData(vec2(x+1,y))>0) return true;  // wall-climb
        for (var k=1; k<=window._JUMP_REACH; k++) {
            var t=getTileCollisionData(vec2(x, y-k));
            if (t>0 || t===tileType_ladder) return true;   // solid ground / ladder to launch from
        }
        return false;
    };

    // ── dig-aware geodesic BFS (mirrors game_server_workers.js) ───────────────
    // Destructible terrain is passable-with-cost = real bullets-to-breach, and
    // digging is HORIZONTAL only (gun can't shoot up/down). Used for the v2 obs.
    window._bfsFrom = function(sx, sy, cx, cy, R) {
        var SZ=2*R+1, minx=cx-R, miny=cy-R, maxx=cx+R, maxy=cy+R;
        var dd=new Array(SZ*SZ); for (var _i=0;_i<dd.length;_i++) dd[_i]=-1;
        function gi(x,y){ return (y-miny)*SZ+(x-minx); }
        function cost(fx, fy, tx, ty){
            var t=getTileCollisionData(vec2(tx,ty));
            if (t<=0 || t===tileType_ladder) return 1;   // walk / climb / fall
            if (tx===fx) return -1;                        // pure vertical into rock: can't dig up/down
            if (t===tileType_glass) return 1;              // shoot through window (horizontal)
            if (t===tileType_dirt) return 5;               // dig dirt
            return 20;                                       // dig base/pipe/solid
        }
        var base={dd:dd,SZ:SZ,minx:minx,miny:miny,maxx:maxx,maxy:maxy,gi:gi,ok:false};
        if (sx<minx||sx>maxx||sy<miny||sy>maxy) return base;
        var MAXC=96, buckets={};
        var s0=gi(sx,sy); dd[s0]=0; buckets[0]=[s0];
        for (var dcur=0; dcur<=MAXC; dcur++) {
            var bk=buckets[dcur]; if (!bk) continue;
            for (var bi=0; bi<bk.length; bi++) {
                var idx=bk[bi];
                if (dd[idx]!==dcur) continue;
                var x=minx+(idx%SZ), y=miny+((idx/SZ)|0);
                for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
                    if (!ax && !ay) continue;
                    var nx=x+ax, ny=y+ay;
                    if (nx<minx||nx>maxx||ny<miny||ny>maxy) continue;
                    var nidx=gi(nx,ny);
                    var c=cost(x,y,nx,ny);
                    if (c<0) continue;
                    if (ny>y && getTileCollisionData(vec2(nx,ny))===tileType_empty && !window._canAscendTo(nx,ny)) continue;  // can't jump that high through open air
                    var nd=dcur+c;
                    if (nd<=MAXC && (dd[nidx]===-1 || nd<dd[nidx])) {
                        dd[nidx]=nd;
                        (buckets[nd]||(buckets[nd]=[])).push(nidx);
                    }
                }
            }
        }
        base.ok=true; return base;
    };

    // ── COARSE global flow-field (mirrors game_server_workers.js) ─────────────
    // Low-res whole-level BFS (4-tile cells, cached per level) to route the long
    // way around rock the local field can't see. Used when the fine field fails.
    window._COARSE_CELL = 4;
    window._buildCoarseMap = function() {
        var CC = window._COARSE_CELL;
        var lx = levelSize ? Math.ceil(levelSize.x) : 200;
        var ly = levelSize ? Math.ceil(levelSize.y) : 100;
        var CW = Math.max(1, Math.ceil(lx/CC)), CH = Math.max(1, Math.ceil(ly/CC));
        var cost = new Array(CW*CH);
        for (var cy=0; cy<CH; cy++) for (var cx=0; cx<CW; cx++) {
            var open=0, dirt=0;
            for (var ty=0; ty<CC; ty++) for (var tx=0; tx<CC; tx++) {
                var t=getTileCollisionData(vec2(cx*CC+tx, cy*CC+ty));
                if (t<=0 || t===tileType_ladder || t===tileType_glass) open++;
                else if (t===tileType_dirt) dirt++;
            }
            cost[cy*CW+cx] = open>0 ? 1 : (dirt>0 ? 4 : -1);
        }
        return {cost:cost, CW:CW, CH:CH};
    };
    window._coarseField = function(px, py, ex, ey) {
        var res = {dist:1.0, dx:0, dy:0, ok:false};
        if (!levelSize) return res;
        var CC = window._COARSE_CELL, NORM = 120, MAXC = 240;
        if (window._coarseLevel !== level) { window._coarseMap = window._buildCoarseMap(); window._coarseLevel = level; }
        var cm = window._coarseMap, CW = cm.CW, CH = cm.CH, cost = cm.cost;
        var sx=Math.floor(px/CC), sy=Math.floor(py/CC), tx=Math.floor(ex/CC), ty=Math.floor(ey/CC);
        if (sx<0||sx>=CW||sy<0||sy>=CH||tx<0||tx>=CW||ty<0||ty>=CH) return res;
        var N=CW*CH, dd=new Array(N); for (var i=0;i<N;i++) dd[i]=-1;
        var buckets={}; dd[sy*CW+sx]=0; buckets[0]=[sy*CW+sx];
        for (var dcur=0; dcur<=MAXC; dcur++) {
            var bk=buckets[dcur]; if (!bk) continue;
            for (var bi=0; bi<bk.length; bi++) {
                var idx=bk[bi]; if (dd[idx]!==dcur) continue;
                var x=idx%CW, y=(idx/CW)|0;
                for (var ax=-1;ax<=1;ax++) for (var ay=-1;ay<=1;ay++) {
                    if (!ax && !ay) continue;
                    var nx=x+ax, ny=y+ay;
                    if (nx<0||nx>=CW||ny<0||ny>=CH) continue;
                    var c=cost[ny*CW+nx]; if (c<0) continue;
                    var nidx=ny*CW+nx, nd=dcur+c;
                    if (nd<=MAXC && (dd[nidx]===-1 || nd<dd[nidx])) { dd[nidx]=nd; (buckets[nd]||(buckets[nd]=[])).push(nidx); }
                }
            }
        }
        var ed=dd[ty*CW+tx];
        if (ed<0) return res;
        res.ok=true; res.dist=Math.min(1, ed/NORM);
        var bx=tx, by=ty, guard=4000;
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
    };

    window._getGameState = function() {
        try {
            const p = players && players[0];
            if (!p) return null;

            const lx = levelSize ? levelSize.x : 200;
            const ly = levelSize ? levelSize.y : 100;
            const facing = p.mirror ? -1 : 1;   // gun fires horizontally this way

            // ── enemies (nearest 10) with shootable + aiming + velocity ───────
            const enemies = [];
            for (const o of engineCollideObjects) {
                if (o.isCharacter && o.team === team_enemy && !o.isDead()) {
                    const dx = o.pos.x - p.pos.x;
                    const dy = o.pos.y - p.pos.y;
                    const sh  = (Math.sign(dx) === facing && Math.abs(dx) < 8 && Math.abs(dy) < 1.2) ? 1 : 0;
                    const aim = (Math.sign(-dx) === (o.mirror?-1:1)) ? (o.holdingShoot?1.0:0.4) : 0.0;
                    enemies.push({ x: dx/lx, y: dy/ly, health: o.health/5,
                                   vx:(o.velocity?o.velocity.x:0)/0.2, vy:(o.velocity?o.velocity.y:0)/0.3,
                                   shootable: sh, aiming: aim, dist: dx*dx+dy*dy });
                }
            }
            enemies.sort((a, b) => a.dist - b.dist);

            // ── 13x9 tile grid centred on player ─────────────────────────────
            const GW = 13, GH = 9, HW = 6, HH = 4;
            const px = Math.round(p.pos.x);
            const py = Math.round(p.pos.y);
            const grid     = new Array(GW * GH).fill(0);   // v1 binary terrain
            const gTerrain = new Array(GW * GH).fill(0);   // v2 dig-cost terrain
            const gEnemy   = new Array(GW * GH).fill(0);
            const gHazard  = new Array(GW * GH).fill(0);

            for (let dy = -HH; dy <= HH; dy++) {
                for (let dx = -HW; dx <= HW; dx++) {
                    const tile = getTileCollisionData(vec2(px+dx, py+dy));
                    let val = 0, dval = 0;
                    if      (tile === tileType_ladder) { val = -1.0; dval = -1.0; }
                    else if (tile === tileType_glass)  { val =  0.5; dval =  0.25; }
                    else if (tile === tileType_dirt)   { val =  1.0; dval =  0.5;  }
                    else if (tile > 0)                 { val =  1.0; dval =  1.0;  }
                    const gidx = (dy+HH)*GW + (dx+HW);
                    grid[gidx] = val; gTerrain[gidx] = dval;
                }
            }

            // Overlay props — explosive ones are tactically important
            for (const o of engineCollideObjects) {
                if (o.isGameObject && !o.isCharacter &&
                    !o.isWeapon && !o.isCheckpoint && !o.destroyed) {
                    const dx = Math.round(o.pos.x - px);
                    const dy = Math.round(o.pos.y - py);
                    if (Math.abs(dx) <= HW && Math.abs(dy) <= HH) {
                        grid[(dy+HH)*GW + (dx+HW)]    = o.explosionSize > 0 ? 0.75 : 0.3;
                        gHazard[(dy+HH)*GW + (dx+HW)] = o.explosionSize > 0 ? 1.0  : 0.5;
                    }
                }
            }
            for (const qe of engineCollideObjects) {        // enemy occupancy channel
                if (qe.isCharacter && qe.team === team_enemy && !qe.isDead()) {
                    const dx = Math.round(qe.pos.x - px), dy = Math.round(qe.pos.y - py);
                    if (Math.abs(dx) <= HW && Math.abs(dy) <= HH)
                        gEnemy[(dy+HH)*GW + (dx+HW)] = Math.max(0.2, qe.health/5);
                }
            }

            // ── nearest enemy bullets (for the vector) + hazard channel ───────
            const bullets = [];
            for (const bb of engineCollideObjects) {
                if (typeof Bullet !== 'undefined' && bb instanceof Bullet && bb.team === team_enemy) {
                    const bdx = bb.pos.x - p.pos.x, bdy = bb.pos.y - p.pos.y;
                    const rdx = Math.round(bb.pos.x - px), rdy = Math.round(bb.pos.y - py);
                    if (Math.abs(rdx) <= HW && Math.abs(rdy) <= HH) gHazard[(rdy+HH)*GW + (rdx+HW)] = 1.0;
                    bullets.push({ x: bdx/lx, y: bdy/ly,
                                   vx:(bb.velocity?bb.velocity.x:0)/0.5, vy:(bb.velocity?bb.velocity.y:0)/0.5,
                                   dist: bdx*bdx+bdy*bdy });
                }
            }
            bullets.sort((a, b) => a.dist - b.dist);

            // ── geodesic flow field: reach channel + geo_dist + geo_dir ───────
            const R = 24, GEO_NORM = 48, REACH_NORM = 16;
            const gReach = new Array(GW * GH).fill(1.0);
            let geoDist = 1.0, geoDx = 0, geoDy = 0;
            const fld = window._bfsFrom(px, py, px, py, R);
            if (fld.ok) {
                if (enemies.length) {
                    const ne = enemies[0];
                    const etx = Math.round(p.pos.x + ne.x*lx), ety = Math.round(p.pos.y + ne.y*ly);
                    if (etx>=fld.minx && etx<=fld.maxx && ety>=fld.miny && ety<=fld.maxy) {
                        const ed = fld.dd[fld.gi(etx,ety)];
                        if (ed >= 0) {
                            geoDist = Math.min(1, ed/GEO_NORM);
                            let bx = etx, by = ety, guard = 4000;
                            while (guard-- > 0) {
                                const cd = fld.dd[fld.gi(bx,by)];
                                if (cd <= 0) break;
                                let best = cd, bnx = bx, bny = by;
                                for (let ax=-1;ax<=1;ax++) for (let ay=-1;ay<=1;ay++) {
                                    if (!ax && !ay) continue;
                                    const nx = bx+ax, ny = by+ay;
                                    if (nx<fld.minx||nx>fld.maxx||ny<fld.miny||ny>fld.maxy) continue;
                                    const v = fld.dd[fld.gi(nx,ny)];
                                    if (v>=0 && v<best) { best=v; bnx=nx; bny=ny; }
                                }
                                if (bnx===bx && bny===by) break;
                                if (best===0) break;
                                bx = bnx; by = bny;
                            }
                            geoDx = Math.sign(bx-px); geoDy = Math.sign(by-py);
                        }
                    }
                }
                for (let wy=-HH;wy<=HH;wy++) for (let wx=-HW;wx<=HW;wx++) {
                    const tx = px+wx, ty = py+wy; let val = 1.0;
                    if (tx>=fld.minx&&tx<=fld.maxx&&ty>=fld.miny&&ty<=fld.maxy) {
                        const dv = fld.dd[fld.gi(tx,ty)];
                        if (dv>=0) val = Math.min(1, dv/REACH_NORM);
                    }
                    gReach[(wy+HH)*GW + (wx+HW)] = val;
                }
            }
            // Long-range routing: coarse whole-level field when the fine field fails;
            // straight-line only as a last resort if even the coarse field finds none.
            var coarseDist = 1.0;
            if (geoDx===0 && geoDy===0 && enemies.length) {
                var _ne0 = enemies[0];
                var _cf = window._coarseField(px, py, p.pos.x + _ne0.x*lx, p.pos.y + _ne0.y*ly);
                if (_cf.ok && (_cf.dx!==0 || _cf.dy!==0)) {
                    geoDx = _cf.dx; geoDy = _cf.dy; coarseDist = _cf.dist;
                } else {
                    geoDx = Math.sign(_ne0.x); geoDy = Math.sign(_ne0.y);
                }
            }

            // ── coarse egocentric global map (3x8x8): terrain, now, memory ────
            const GG = 8, CELL = 8, half = GG/2;
            const gTerr = new Array(GG*GG).fill(0), gNow = new Array(GG*GG).fill(0);
            for (let cyi=0;cyi<GG;cyi++) for (let cxi=0;cxi<GG;cxi++) {
                const sx = Math.round(p.pos.x + (cxi-half+0.5)*CELL);
                const sy = Math.round(p.pos.y + (cyi-half+0.5)*CELL);
                const tt = getTileCollisionData(vec2(sx,sy));
                gTerr[cyi*GG+cxi] = (tt>0 && tt!==tileType_ladder) ? 1 : 0;
            }
            for (const ge of engineCollideObjects) {
                if (ge.isCharacter && ge.team===team_enemy && !ge.isDead()) {
                    const gcx = Math.floor((ge.pos.x-p.pos.x)/CELL)+half;
                    const gcy = Math.floor((ge.pos.y-p.pos.y)/CELL)+half;
                    if (gcx>=0&&gcx<GG&&gcy>=0&&gcy<GG) gNow[gcy*GG+gcx] = Math.min(1, gNow[gcy*GG+gcx]+0.5);
                }
            }
            if (window._gmemLevel !== level) { window._gmem = new Array(GG*GG).fill(0); window._gmemLevel = level; }
            const gm = window._gmem;
            for (let mi=0; mi<GG*GG; mi++) gm[mi] = Math.min(1, gm[mi]*0.97 + gNow[mi]);
            const globe = gTerr.concat(gNow).concat(gm.slice());

            return {
                px:          p.pos.x / lx,
                py:          p.pos.y / ly,
                vx:          p.velocity.x / 0.2,
                vy:          p.velocity.y / 0.3,
                health:      p.health,
                ground:      p.groundTimer.active() ? 1.0 : 0.0,
                grenades:    (p.grenadeCount || 0) / 3.0,
                dodge_ready: !p.dodgeRechargeTimer.active() ? 1.0 : 0.0,
                on_fire:     p.burnTimer.active() ? 1.0 : 0.0,
                alive:       !p.isDead(),
                lives:       playerLives,
                kills:       totalKills,
                level:       level,
                warmup:      levelWarmup ? 1 : 0,
                enemies:     enemies.slice(0, 10).map(e => [e.x, e.y, e.health]),
                enemy_vels:  enemies.slice(0, 10).map(e => [e.vx, e.vy]),
                enemies_remaining: (typeof levelEnemyCount!=='undefined'?Math.max(0,levelEnemyCount):0)+enemies.length,
                facing:      facing,
                enemy_shootable: enemies.slice(0, 10).map(e => e.shootable),
                bullets:     bullets.slice(0, 5).map(b => [b.x, b.y, b.vx, b.vy]),
                n_enemies:   enemies.length,
                grid:        grid,
                grid_terrain: gTerrain, grid_enemy: gEnemy, grid_hazard: gHazard,
                grid_reach:  gReach, geo_dist: geoDist, geo_dir: [geoDx, geoDy],
                coarse_dist: coarseDist,
                enemy_aiming: enemies.slice(0, 10).map(e => e.aiming),
                globe:       globe,
                fps:         typeof averageFPS !== 'undefined' ? averageFPS : -1,
            };
        } catch (e) {
            return null;
        }
    };

    window._resetGame = () => { window._coarseLevel = null; resetGame(); };

    // Manual state-dump trigger: press 'L' in the game window to flag a dump.
    // watch_game.py polls window._dumpReq and writes the full state to a log,
    // so a stuck spot can be captured and inspected exactly.
    if (!window._dumpHooked) {
        window._dumpHooked = 1;
        window._dumpReq = 0;
        window.addEventListener('keydown', function(e){
            if (e.key === 'l' || e.key === 'L') window._dumpReq = (window._dumpReq||0) + 1;
        });
    }

    console.log('[AI] helpers injected');
})();
"""

N_ENEMIES  = 5
GRID_W, GRID_H = 13, 9
OBS_DIM    = 9 + N_ENEMIES * 3 + GRID_W * GRID_H   # 9 + 15 + 117 = 141


class SpaceHuggersEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self,
                 game_path: str,
                 headless: bool = True,
                 frame_ms: int = 50,
                 action_repeat: int = 1,
                 restart_every: int = 200,
                 obs_version: int = 1):
        """
        game_path      : path to SpaceHuggers-main/
        headless       : run browser without a window
        frame_ms       : ms to hold each action (50 ms ≈ 3 game frames at 60fps)
        action_repeat  : repeat each action this many sub-frames (1 = no repeat)
        restart_every  : restart browser every N episodes (prevents memory leaks)
        obs_version    : 1 = flat 141-float Box; 2 = Tier-2 Dict (grid+globe+vector)
        """
        super().__init__()
        self.game_path      = Path(game_path).resolve()
        self.headless       = headless
        self.frame_ms       = frame_ms
        self.restart_every  = restart_every
        self.obs_version    = obs_version

        # [horizontal, vertical, shoot, dodge, grenade]
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2])
        if obs_version == 2:
            self.observation_space = gym.spaces.Dict({
                "grid":   gym.spaces.Box(-5.0, 5.0, (GRID_CH_V2, V2_GRID_H, V2_GRID_W), np.float32),
                "globe":  gym.spaces.Box(-5.0, 5.0, (GLOBE_CH, GLOBE_SZ, GLOBE_SZ), np.float32),
                "vector": gym.spaces.Box(-5.0, 5.0, (VEC_DIM_V2,), np.float32),
            })
            self._obs_fn = _to_obs_v2          # module fn: dict state -> Dict obs
        else:
            self.observation_space = gym.spaces.Box(
                low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self._obs_fn = self._to_obs        # bound method: dict state -> flat obs

        self.action_repeat  = action_repeat
        self._pw = self._browser = self._page = None

        # Episode state
        self._last_kills      = 0
        self._last_lives      = 10   # game starts with 10 lives
        self._last_level      = 1
        self._last_enemy_dist = None  # for approach reward
        self._step_n          = 0
        self._episode_n       = 0

    # ── browser lifecycle ────────────────────────────────────────────────────

    def _launch(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass

        self._pw      = sync_playwright().start()
        args = ["--mute-audio"]
        if self.headless:
            args += ["--use-gl=swiftshader"]
        self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
        self._page    = self._browser.new_page(viewport={"width": 800, "height": 600})
        self._page.goto(self.game_path.as_uri() + "/index.html")
        time.sleep(2.5)
        self._page.evaluate(_INJECT)
        time.sleep(0.2)

    def _ensure_started(self):
        if self._page is None:
            self._launch()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _raw(self):
        try:
            return self._page.evaluate("window._getGameState()")
        except Exception:
            return None

    def _to_obs(self, s) -> np.ndarray:
        if s is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        obs = [
            s["px"], s["py"], s["vx"], s["vy"],
            s["health"], s["ground"],
            s["grenades"], s["dodge_ready"], s["on_fire"],
        ]
        for i in range(N_ENEMIES):
            if i < len(s["enemies"]):
                obs.extend(s["enemies"][i])
            else:
                obs.extend([0.0, 0.0, 0.0])
        obs.extend(s.get("grid") or [0.0] * (GRID_W * GRID_H))
        return np.clip(np.array(obs, dtype=np.float32), -5.0, 5.0)

    def _send(self, action):
        m, v, sh, dg, gr = (int(x) for x in action)
        left  = 1 if m == 1 else 0
        right = 1 if m == 2 else 0
        up    = 1 if v == 1 else 0
        down  = 1 if v == 2 else 0
        self._page.evaluate(
            f"window._setAIInput({left},{right},{up},{down},{sh},{dg},{gr})"
        )

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_n += 1

        if self.restart_every and self._episode_n % self.restart_every == 0:
            print(f"  [env] restarting browser at episode {self._episode_n}")
            self._launch()
        else:
            self._ensure_started()

        self._page.evaluate("window._resetGame()")
        time.sleep(2.8)   # level warmup is 2 s; extra buffer for safety

        self._last_kills      = 0
        self._last_lives      = 10
        self._last_level      = 1
        self._last_enemy_dist = None
        self._step_n          = 0

        return self._obs_fn(self._raw()), {}

    def step(self, action):
        # ── Action repeat ─────────────────────────────────────────────────────
        # Hold the same action for `action_repeat` sub-frames (default 4 × 50 ms
        # = 200 ms per decision).  Benefits:
        #   • Actions look deliberate instead of randomly flickering every 50 ms
        #   • Longer commitment window improves temporal credit assignment
        #   • Same 25-min wall-clock budget (7 500 steps × 200 ms)
        total_reward = 0.0
        obs          = self._obs_fn(None)      # zero obs (flat array or Dict per obs_version)
        info         = {}
        terminated   = False
        truncated    = False

        for _ in range(self.action_repeat):
            self._send(action)
            time.sleep(self.frame_ms / 1000.0)

            s   = self._raw()
            obs = self._obs_fn(s)

            kills     = s["kills"]     if s else 0
            lives     = s["lives"]     if s else self._last_lives
            alive     = s["alive"]     if s else True
            level     = s["level"]     if s else self._last_level
            n_enemies = s["n_enemies"] if s else 0

            delta_kills = kills - self._last_kills
            delta_lives = max(0, self._last_lives - lives)   # lives only count down
            delta_level = level - self._last_level

            # ── Primary rewards ───────────────────────────────────────────────
            kill_r  = delta_kills * 10.0    # +10 per enemy killed
            death_p = delta_lives * 5.0     # -5 per life lost
            level_r = delta_level * 150.0   # +150 per level completed

            # ── Dense navigation: approach reward ─────────────────────────────
            # Clipped to ±0.05/step so it shapes without overwhelming kill reward.
            # Skipped on kill steps (nearest enemy changes → dist delta invalid).
            ey = 0.0   # nearest enemy Y — initialised here so ladder block reads it
            approach_r = 0.0
            if n_enemies > 0 and s and s.get("enemies"):
                ex, ey, _ = s["enemies"][0]
                dist_now = (ex*ex + ey*ey) ** 0.5
                if self._last_enemy_dist is not None and delta_kills == 0:
                    d_delta    = self._last_enemy_dist - dist_now
                    approach_r = max(min(d_delta * 2.0, 0.05), -0.05)
                self._last_enemy_dist = dist_now
            else:
                self._last_enemy_dist = None

            # ── Mechanic bonuses ──────────────────────────────────────────────
            dodge_act = int(action[3])
            vert_act  = int(action[1])
            on_fire   = s.get("on_fire", 0) if s else 0
            on_ground = bool(s.get("ground", 1)) if s else True

            fire_r = 0.1 if (on_fire and dodge_act) else 0.0

            grid        = s.get("grid", []) if s else []
            on_ladder   = len(grid) > 58 and grid[58] < -0.5
            enemy_above = n_enemies > 0 and ey > 0.05
            if on_ladder and vert_act != 0:
                ladder_r = 0.015 if enemy_above else 0.003
            else:
                ladder_r = 0.0

            # air_dodge_r was causing jump+dodge spam worth ~+90/episode — removed.
            air_dodge_r = 0.0

            wall_left   = len(grid) > 57 and grid[57] > 0.4
            wall_right  = len(grid) > 59 and grid[59] > 0.4
            horiz_act   = int(action[0])
            shoot_act   = int(action[2])
            pressing_into_wall = (wall_left  and horiz_act == 1) or \
                                 (wall_right and horiz_act == 2)
            wall_climb_r = 0.01 if (pressing_into_wall and not on_ground
                                    and not on_ladder and vert_act == 1
                                    and shoot_act == 0) else 0.0

            survive = 0.0   # removed +0.002/step survival bonus — it fueled turtling

            reward = (kill_r - death_p + level_r + approach_r
                      + fire_r + ladder_r + wall_climb_r + air_dodge_r + survive)

            self._last_kills  = kills
            self._last_lives  = lives
            self._last_level  = level
            self._step_n     += 1
            total_reward     += reward

            terminated = (lives <= 0 and not alive)
            # 30 000 steps × 50 ms = 25 min max — enough to complete several levels
            truncated  = self._step_n > 30_000

            info = {"kills": kills, "lives": lives, "level": level}

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._page = None
