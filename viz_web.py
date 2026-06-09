"""
Force-directed ("spiderweb") visualisation of the trained policies.

Outputs neuron_web.html — self-contained, interactive:
  - live physics layout: neurons repel, connections pull -> organic web
  - drag a neuron (it pins; double-click background to unpin all)
  - pan (drag background), zoom (wheel)
  - hover to highlight a neuron's links
  - toggle: Web (force) <-> Radial (concentric rings by layer)
  - switch between models v1 / v2
Edges coloured by weight sign (blue +, red -), opacity by magnitude.
"""
import zipfile, io, os, glob, json
import numpy as np
import torch

MAXN = 18  # neurons shown per layer (downsampled)


def load_sd(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read("policy.pth")
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location="cpu")
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)


def npw(sd, k):
    return sd[k].detach().cpu().numpy()


MODELS = {}
if os.path.exists("game_models/ppo_sh_latest.zip"):
    MODELS["v1"] = "game_models/ppo_sh_latest.zip"
v2g = sorted(glob.glob("game_models_v2_pre_perception_*/ppo_sh_bestlevel.zip"))
if v2g:
    MODELS["v2"] = v2g[-1]
SD = {k: load_sd(p) for k, p in MODELS.items()}


def pick(n):
    return list(range(n)) if n <= MAXN else [int(round(i*(n-1)/(MAXN-1))) for i in range(MAXN)]


def spec(sd, kind):
    if kind == "v1":
        layers = [("input", 141), ("hidden", 256), ("hidden", 256), ("actions", 12)]
        keys = ["mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight", "action_net.weight"]
    else:
        feat = sd["mlp_extractor.policy_net.0.weight"].shape[1]
        layers = [("features", feat), ("hidden", 512), ("hidden", 768), ("hidden", 512), ("actions", 12)]
        keys = ["mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight",
                "mlp_extractor.policy_net.4.weight", "action_net.weight"]
    W = [npw(sd, k) for k in keys]
    shown = [pick(n) for _, n in layers]
    blocks, maxw = [], 1e-9
    for li in range(len(layers)-1):
        w = W[li]; oi, ii = shown[li+1], shown[li]
        blocks.append([[float(w[o, i]) for i in ii] for o in oi])
        maxw = max(maxw, float(np.abs(w[np.ix_(oi, ii)]).max()))
    return {"name": kind,
            "layers": [{"label": lab, "size": n, "shown": len(s)} for (lab, n), s in zip(layers, shown)],
            "blocks": blocks, "maxw": maxw}


specs = {k: spec(SD[k], k) for k in SD}

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>SpaceHuggers policy — spiderweb</title>
<style>
 html,body{margin:0;height:100%;background:#0b0e14;color:#cdd9e5;font:13px system-ui,Segoe UI,Arial}
 #bar{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:#11151ccc;
   border-bottom:1px solid #232a36;backdrop-filter:blur(4px);z-index:5}
 button{background:#1b2230;color:#cdd9e5;border:1px solid #2b3445;border-radius:6px;
   padding:6px 11px;margin-right:6px;cursor:pointer}
 button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 #hint{color:#6b7686}
 #tip{position:fixed;pointer-events:none;background:#11151ce6;border:1px solid #2b3445;
   border-radius:6px;padding:3px 7px;font-size:12px;display:none;z-index:6}
 canvas{display:block;cursor:grab}
</style></head><body>
<div id="bar">
 <span id="models"></span> &nbsp;|&nbsp;
 <button id="modeWeb" class="on">web (force)</button>
 <button id="modeRad">radial</button>
 <button id="reheat">reheat</button>
 <span id="hint"> drag a neuron to pull the web · drag background = pan · wheel = zoom ·
  <span style="color:#58a6ff">+</span>/<span style="color:#f85149">−</span> weight</span>
</div>
<canvas id="c"></canvas><div id="tip"></div>
<script>
const DATA = __DATA__;
let cur = Object.keys(DATA)[0], mode='web';
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
let view={x:0,y:0,s:1}, drag=null, dragNode=null, hover=null;
let nodes=[], edges=[], temp=1;

function build(){
  const g=DATA[cur]; nodes=[]; edges=[];
  const cx=cv.width/2, cy=cv.height/2;
  const idx=(li,i)=>{ // global node index
    let c=0; for(let l=0;l<li;l++) c+=g.layers[l].shown; return c+i; };
  for(let li=0; li<g.layers.length; li++){
    const k=g.layers[li].shown;
    for(let i=0;i<k;i++){
      const ang=Math.random()*7, r=40+Math.random()*Math.min(cx,cy)*0.6;
      nodes.push({li,i,label:g.layers[li].label,size:g.layers[li].size,shown:k,
        x:cx+Math.cos(ang)*r, y:cy+Math.sin(ang)*r, vx:0, vy:0, pin:false});
    }
  }
  for(let bi=0;bi<g.blocks.length;bi++){const blk=g.blocks[bi];
    for(let o=0;o<blk.length;o++)for(let inp=0;inp<blk[o].length;inp++){
      const w=blk[o][inp]; if(Math.abs(w)/g.maxw<0.05)continue;
      edges.push({a:idx(bi,inp), b:idx(bi+1,o), w});
    }}
  temp=1;
}
function radial(){ const g=DATA[cur], cx=cv.width/2, cy=cv.height/2, R=Math.min(cx,cy)-70;
  const L=g.layers.length;
  nodes.forEach(n=>{ const ring=(n.li+0.6)/L*R; const k=n.shown;
    const a=(n.i/k)*2*Math.PI - Math.PI/2;
    n.x=cx+Math.cos(a)*ring; n.y=cy+Math.sin(a)*ring; n.vx=n.vy=0; });
}
function step(){
  if(mode!=='web'||temp<0.02) return;
  const REP=2600, SPR=0.012, REST=70, G=0.008, cx=cv.width/2, cy=cv.height/2;
  for(let i=0;i<nodes.length;i++){const a=nodes[i]; if(a.pin)continue;
    let fx=(cx-a.x)*G, fy=(cy-a.y)*G;
    for(let j=0;j<nodes.length;j++){ if(i===j)continue; const b=nodes[j];
      let dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy+0.01, d=Math.sqrt(d2);
      const f=REP/d2; fx+=dx/d*f; fy+=dy/d*f; }
    a._fx=fx; a._fy=fy;
  }
  for(const e of edges){ const a=nodes[e.a], b=nodes[e.b];
    let dx=b.x-a.x, dy=b.y-a.y, d=Math.hypot(dx,dy)+0.01;
    const f=SPR*(d-REST); const ux=dx/d*f, uy=dy/d*f;
    if(!a.pin){a._fx+=ux; a._fy+=uy;} if(!b.pin){b._fx-=ux; b._fy-=uy;}
  }
  for(const a of nodes){ if(a.pin)continue;
    a.vx=(a.vx+a._fx)*0.85; a.vy=(a.vy+a._fy)*0.85;
    a.x+=a.vx*temp; a.y+=a.vy*temp; }
  temp*=0.992;
}
const T=(x)=>x*view.s+view.x, Ty=(y)=>y*view.s+view.y;
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height); const g=DATA[cur];
  for(const e of edges){ const a=nodes[e.a], b=nodes[e.b];
    const al=Math.min(1,Math.abs(e.w)/g.maxw);
    const hot=hover&&(e.a===hover||e.b===hover);
    ctx.strokeStyle=(e.w>=0?(hot?'rgba(88,166,255,':'rgba(56,110,190,'):(hot?'rgba(248,81,73,':'rgba(170,64,60,'))+(hot?Math.min(1,al+0.35):al*0.5)+')';
    ctx.lineWidth=(hot?1.7:0.6)*Math.max(0.4,al);
    ctx.beginPath(); ctx.moveTo(T(a.x),Ty(a.y)); ctx.lineTo(T(b.x),Ty(b.y)); ctx.stroke(); }
  const pal=['#e3b341','#7ee787','#79c0ff','#d2a8ff','#ff7b72'];
  for(let n=0;n<nodes.length;n++){const a=nodes[n], hot=hover===n;
    ctx.beginPath(); ctx.arc(T(a.x),Ty(a.y),(hot?7:4.5)*Math.max(0.6,view.s),0,7);
    ctx.fillStyle=hot?'#fff':pal[a.li%pal.length]; ctx.fill();
    if(hot){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();}}
}
function loop(){ step(); draw(); requestAnimationFrame(loop); }

function pick(mx,my){ let best=-1,bd=15;
  for(let n=0;n<nodes.length;n++){const d=Math.hypot(T(nodes[n].x)-mx,Ty(nodes[n].y)-my); if(d<bd){bd=d;best=n;}}
  return best; }
cv.addEventListener('mousedown',e=>{ const n=pick(e.clientX,e.clientY);
  if(n>=0){dragNode=n; nodes[n].pin=true;} else {drag={x:e.clientX-view.x,y:e.clientY-view.y}; cv.style.cursor='grabbing';} });
addEventListener('mouseup',()=>{drag=null; dragNode=null; cv.style.cursor='grab';});
cv.addEventListener('mousemove',e=>{
  if(dragNode!=null && dragNode>=0){ const a=nodes[dragNode];
    a.x=(e.clientX-view.x)/view.s; a.y=(e.clientY-view.y)/view.s; a.vx=a.vy=0; temp=Math.max(temp,0.3); return; }
  if(drag){view.x=e.clientX-drag.x; view.y=e.clientY-drag.y; return;}
  const n=pick(e.clientX,e.clientY); if(n!==hover)hover=n;
  if(n>=0){tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
    tip.textContent=nodes[n].label+' #'+nodes[n].i+(nodes[n].size>nodes[n].shown?' ('+nodes[n].shown+'/'+nodes[n].size+')':'');}
  else tip.style.display='none';
});
cv.addEventListener('dblclick',()=>{nodes.forEach(n=>n.pin=false); temp=Math.max(temp,0.5);});
cv.addEventListener('wheel',e=>{e.preventDefault(); const f=e.deltaY<0?1.1:0.9, mx=e.clientX,my=e.clientY;
  view.x=mx-(mx-view.x)*f; view.y=my-(my-view.y)*f; view.s*=f;},{passive:false});

function resize(){cv.width=innerWidth; cv.height=innerHeight; build();}
addEventListener('resize',resize);
function setMode(m){mode=m; document.getElementById('modeWeb').classList.toggle('on',m==='web');
  document.getElementById('modeRad').classList.toggle('on',m==='radial');
  if(m==='radial') radial(); else temp=1;}
document.getElementById('modeWeb').onclick=()=>setMode('web');
document.getElementById('modeRad').onclick=()=>setMode('radial');
document.getElementById('reheat').onclick=()=>{build(); temp=1;};
const mb=document.getElementById('models');
Object.keys(DATA).forEach(m=>{const b=document.createElement('button'); b.textContent='model '+m; b.dataset.m=m;
  if(m===cur)b.classList.add('on');
  b.onclick=()=>{cur=m; mb.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.m===m)); build(); temp=1;}; mb.appendChild(b);});
view={x:0,y:0,s:1}; resize(); loop();
</script></body></html>"""

with open("neuron_web.html", "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(specs)))
print("wrote neuron_web.html")
