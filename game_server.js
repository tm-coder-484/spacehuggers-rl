/**
 * SpaceHuggers Headless Game Server
 *
 * Runs the game's JavaScript directly in Node.js — no browser, no Chromium.
 * Communicates with the Python RL env via stdin/stdout newline-delimited JSON.
 *
 * Protocol:
 *   → {"type":"reset"}
 *   ← {"type":"state","state":{...}}
 *
 *   → {"type":"step","action":[m,v,sh,dg,gr]}
 *   ← {"type":"state","state":{...}}
 *
 *   → {"type":"ping"}
 *   ← {"type":"pong"}
 *
 * Env vars:
 *   GAME_PATH  path to SpaceHuggers-main directory
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const vm   = require('vm');

const GAME_DIR = process.env.GAME_PATH
    || path.join(__dirname, 'SpaceHuggers-main');

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Browser API mocks (set on global BEFORE loading game code)
// ─────────────────────────────────────────────────────────────────────────────

// RAF: capture the callback — we drive frames manually
global._rafCallback = null;
global._frameCount  = 0;
global.requestAnimationFrame = (fn) => { global._rafCallback = fn; };

// window = global (game uses window['chrome'], window.innerWidth, etc.)
global.window      = global;
global.innerWidth  = 800;
global.innerHeight = 600;
global.chrome      = 1;   // ⟹ lowGraphicsSettings=0, glOverlay=0 in app.js

// Image — tile sheet is base64-embedded; fire onload synchronously
global.Image = class FakeImage {
    set src(_) {
        // Tile sheet is 128×64 px (embedded PNG in engine.js)
        this.width  = 128;
        this.height = 64;
        if (typeof this.onload === 'function') this.onload();
    }
};

// Deep-noop object: any method call returns this same proxy (enabling chaining
// like ctx.createLinearGradient().addColorStop() without crashing).
function makeDeepNoop() {
    const store = {};
    const proxy = new Proxy(store, {
        get(t, k) {
            if (k in t) return t[k];
            if (typeof k === 'symbol') return undefined;
            return (..._args) => proxy;   // any method call → return proxy itself
        },
        set(t, k, v) { t[k] = v; return true; },
    });
    return proxy;
}
const makeNoopCtx = makeDeepNoop;   // canvas 2D context is just a deep-noop

// Fake canvas element (returned by document.createElement('canvas'))
function makeCanvas(w = 800, h = 600) {
    const ctx2d = makeNoopCtx();
    const el = {
        width:  w,
        height: h,
        style:  {},
        getContext(type) { return type === '2d' ? ctx2d : null; },
        getBoundingClientRect() { return { left:0, right:w, top:0, bottom:h, width:w, height:h }; },
    };
    ctx2d.canvas = el;
    return el;
}

// document
global.document = {
    createElement(tag) {
        if (tag === 'canvas') return makeCanvas();
        return { style: {}, appendChild() {}, tagName: tag.toUpperCase(),
                 innerHTML: '', disabled: false, value: '', type: '',
                 oninput: null, onclick: null };
    },
    createTextNode() { return {}; },
    body: {
        appendChild() {},
        style: '',
    },
    hasFocus() { return true; },
};

// Audio — full no-op; game code only creates AudioContext lazily on first sound,
// and we patch soundEnable=0 so no sound is ever played.
global.AudioContext = class FakeAudioContext {
    constructor()          {}
    get currentTime()      { return global._frameCount / 60; }
    createBuffer()         { return { getChannelData: () => new Float32Array(0) }; }
    createBufferSource()   { return { buffer:null, loop:false, connect(){}, start(){}, disconnect(){} }; }
    createGain()           { return { gain:{value:1}, connect(){} }; }
    get destination()      { return {}; }
    decodeAudioData(b, ok) { ok && ok({}); }
};
global.speechSynthesis        = { speak() {}, cancel() {} };
global.SpeechSynthesisUtterance = class {};

// Input event handlers (assigned as globals by engineInput.js)
global.onkeydown = global.onkeyup = global.onmousedown = global.onmouseup =
global.onmousemove = global.oncontextmenu = global.onwheel =
global.ontouchstart = global.ontouchmove = global.ontouchend = null;

// Gamepad API — navigator is read-only in Node 22, use defineProperty
try {
    Object.defineProperty(global, 'navigator', {
        value:      { getGamepads() { return []; }, userAgent: 'Node.js' },
        writable:   true,
        configurable: true,
    });
} catch (_) {
    // Already writable in older Node versions — ignore
}

// Misc browser globals
global.performance = { now: () => Date.now() };
global.location    = { href: '' };

// Make require available inside the vm context
global.require = require;
global.process = process;
global.console = console;

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Load & concatenate all game source files (with patches)
// ─────────────────────────────────────────────────────────────────────────────

const FILES = [
    // [relative path,  [[searchString, replacement], ...]]
    ['engine/engineUtil.js',   []],
    ['engine/engineDebug.js',  [['const debug = 1',       'const debug = 0'      ]]],
    ['engine/engine.js',       []],
    ['engine/engineObject.js', []],
    ['engine/engineWebGL.js',  [['const glEnable = 1',    'const glEnable = 0'   ]]],
    ['engine/engineDraw.js',   []],
    ['engine/engineInput.js',  []],
    ['engine/engineAudio.js',  [['const soundEnable = 1', 'const soundEnable = 0']]],
    ['engine/engineTileLayer.js', []],
    ['engine/engineParticle.js',  []],
    ['appObjects.js',     []],
    ['appCharacters.js',  []],
    ['appEffects.js',     []],
    ['appLevel.js',       [['const warmUpTime = 2', 'const warmUpTime = 0']]],
    ['app.js',            []],
];

let gameCode = '';
for (const [relPath, patches] of FILES) {
    let src = fs.readFileSync(path.join(GAME_DIR, relPath), 'utf8');
    for (const [from, to] of patches) {
        // Replace first occurrence only (const declarations appear once)
        src = src.replace(from, to);
    }
    // Strip 'use strict' directives so all files share one non-strict scope;
    // this allows `var` declarations to become globals and avoids strict-mode
    // errors from the game's assignment-to-undeclared-var patterns.
    src = src.replace(/'use strict';\s*/g, '').replace(/"use strict";\s*/g, '');
    gameCode += `\n// ═══ ${relPath} ═══\n` + src;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Runtime code appended to game code
//     Has full lexical access to every const/let/var declared in the game files.
// ─────────────────────────────────────────────────────────────────────────────

const RUNTIME = /* javascript */ `

// ── helpers ──────────────────────────────────────────────────────────────────

function _step() {
    if (global._rafCallback) {
        var cb = global._rafCallback;
        global._rafCallback = null;
        global._frameCount++;
        // Advance time by exactly one frame (1000/60 ms) for a stable fixed step.
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
    set(37, left);   // left arrow
    set(39, right);  // right arrow
    set(38, up);     // up arrow / jump / climb
    set(40, down);   // down arrow / descend ladder
    set(90, sh);     // Z = shoot
    set(88, dg);     // X = dodge roll
    set(67, gr);     // C = throw grenade
    isUsingGamepad = 0;
}

function _getGameState() {
    try {
        var p = players && players[0];
        if (!p) return null;

        var lx = levelSize ? levelSize.x : 200;
        var ly = levelSize ? levelSize.y : 100;

        // nearest enemies, sorted by distance
        var enemies = [];
        for (var _ei = 0; _ei < engineCollideObjects.length; _ei++) {
            var o = engineCollideObjects[_ei];
            if (o.isCharacter && o.team === team_enemy && !o.isDead()) {
                var dx = o.pos.x - p.pos.x;
                var dy = o.pos.y - p.pos.y;
                enemies.push({ x: dx/lx, y: dy/ly, health: o.health/5, dist: dx*dx+dy*dy });
            }
        }
        enemies.sort(function(a,b){ return a.dist - b.dist; });

        // 13×9 tile grid centred on player
        var GW = 13, GH = 9, HW = 6, HH = 4;
        var px = Math.round(p.pos.x);
        var py = Math.round(p.pos.y);
        var grid = new Array(GW * GH).fill(0);
        for (var dy = -HH; dy <= HH; dy++) {
            for (var dx = -HW; dx <= HW; dx++) {
                var tile = getTileCollisionData(vec2(px+dx, py+dy));
                var val = 0;
                if      (tile === tileType_ladder) val = -1.0;
                else if (tile === tileType_glass)  val =  0.5;
                else if (tile > 0)                 val =  1.0;
                grid[(dy+HH)*GW + (dx+HW)] = val;
            }
        }
        // overlay props onto grid (explosive ones especially important)
        for (var _oi = 0; _oi < engineCollideObjects.length; _oi++) {
            var oo = engineCollideObjects[_oi];
            if (oo.isGameObject && !oo.isCharacter &&
                !oo.isWeapon && !oo.isCheckpoint && !oo.destroyed) {
                var odx = Math.round(oo.pos.x - px);
                var ody = Math.round(oo.pos.y - py);
                if (Math.abs(odx) <= HW && Math.abs(ody) <= HH) {
                    grid[(ody+HH)*GW + (odx+HW)] = oo.explosionSize > 0 ? 0.75 : 0.3;
                }
            }
        }

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
            enemies:     enemies.slice(0, 5).map(function(e){ return [e.x, e.y, e.health]; }),
            n_enemies:   enemies.length,
            grid:        grid,
        };
    } catch(e) {
        return null;
    }
}

// nextLevel() already runs its warmup synchronously (120 frames of
// engineUpdateObjects inside a JS for-loop) — no extra stepping needed.
// Just run a handful of frames so appUpdate/appRender have run at least once.
for (var _bi = 0; _bi < 5; _bi++) _step();

// ── stdin/stdout JSON protocol ────────────────────────────────────────────────
var readline = require('readline');
var _rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

_rl.on('line', function(rawLine) {
    var line = rawLine.trim();
    if (!line) return;
    var msg;
    try { msg = JSON.parse(line); } catch(e) { return; }

    if (msg.type === 'reset') {
        resetGame();  // nextLevel() inside runs warmup synchronously — game ready immediately
        for (var _ri = 0; _ri < 5; _ri++) _step();  // a few frames for appUpdate/appRender
        process.stdout.write(JSON.stringify({ type: 'state', state: _getGameState() }) + '\\n');
        return;
    }

    if (msg.type === 'step') {
        var a = msg.action;
        _setAIInput(a[0], a[1], a[2], a[3], a[4]);
        _step();
        // Note: engineUpdate already clears p/r flags at end of each tick —
        // no need to clear manually here.
        process.stdout.write(JSON.stringify({ type: 'state', state: _getGameState() }) + '\\n');
        return;
    }

    if (msg.type === 'ping') {
        process.stdout.write(JSON.stringify({ type: 'pong' }) + '\\n');
        return;
    }
});

_rl.on('close', function() { process.exit(0); });

process.stderr.write('[game_server] ready\\n');
`;

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Execute — one vm call so all const/let/var share the same lexical scope
// ─────────────────────────────────────────────────────────────────────────────

const combined = gameCode + '\n' + RUNTIME;

try {
    vm.runInThisContext(combined, {
        filename:    'SpaceHuggers.js',
        lineOffset:  0,
        displayErrors: true,
    });
} catch (err) {
    process.stderr.write('[game_server] FATAL: ' + err.stack + '\n');
    process.exit(1);
}
