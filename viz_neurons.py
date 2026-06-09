"""
Neuron-and-connection visualisation of the trained policies.

Outputs:
  neuron_viz.html  — self-contained interactive node-link diagram (both models,
                     downsampled). Pan (drag), zoom (wheel), hover a neuron to
                     highlight its connections. Edges coloured by weight sign,
                     opacity by magnitude.
  v1_policy.onnx / v2_policy.onnx — reconstructed policy graphs for Netron
                     (drag into https://netron.app).
"""
import zipfile, io, os, glob, json
import numpy as np
import torch
import torch.nn as nn

MAXN = 22  # max neurons shown per layer (downsample for legibility)


def load_sd(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read("policy.pth")
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location="cpu")
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)


def npw(sd, key):
    return sd[key].detach().cpu().numpy()


# ── locate models ──────────────────────────────────────────────────────────────
MODELS = {}
if os.path.exists("game_models/ppo_sh_latest.zip"):
    MODELS["v1"] = "game_models/ppo_sh_latest.zip"
v2g = sorted(glob.glob("game_models_v2_pre_perception_*/ppo_sh_bestlevel.zip"))
if v2g:
    MODELS["v2"] = v2g[-1]
SD = {k: load_sd(p) for k, p in MODELS.items()}


# ── build a downsampled layered graph spec from the policy-path Linear weights ──
def pick(n):
    if n <= MAXN:
        return list(range(n))
    return [int(round(i * (n - 1) / (MAXN - 1))) for i in range(MAXN)]


def graph_spec(sd, kind):
    # ordered list of (label, size) and the weight matrix INTO each non-input layer
    if kind == "v1":
        layers = [("input\n141", 141), ("hidden\n256", 256),
                  ("hidden\n256", 256), ("actions\n12", 12)]
        W = [npw(sd, "mlp_extractor.policy_net.0.weight"),
             npw(sd, "mlp_extractor.policy_net.2.weight"),
             npw(sd, "action_net.weight")]
    else:
        feat = sd["mlp_extractor.policy_net.0.weight"].shape[1]
        layers = [(f"features\n{feat}\n(CNN+vec)", feat), ("hidden\n512", 512),
                  ("hidden\n768", 768), ("hidden\n512", 512), ("actions\n12", 12)]
        W = [npw(sd, "mlp_extractor.policy_net.0.weight"),
             npw(sd, "mlp_extractor.policy_net.2.weight"),
             npw(sd, "mlp_extractor.policy_net.4.weight"),
             npw(sd, "action_net.weight")]
    shown = [pick(n) for _, n in layers]
    blocks, maxw = [], 1e-9
    for li in range(len(layers) - 1):
        w = W[li]                       # shape (out=layers[li+1], in=layers[li])
        out_idx, in_idx = shown[li + 1], shown[li]
        blk = [[float(w[o, i]) for i in in_idx] for o in out_idx]
        blocks.append(blk)
        maxw = max(maxw, np.abs(w[np.ix_(out_idx, in_idx)]).max())
    return {
        "name": kind,
        "layers": [{"label": lab, "size": n, "shown": len(s)}
                   for (lab, n), s in zip(layers, shown)],
        "blocks": blocks,
        "maxw": float(maxw),
    }


specs = {k: graph_spec(SD[k], k) for k in SD}

# ── write the interactive HTML ───────────────────────────────────────────────────
HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>SpaceHuggers policy — neurons & connections</title>
<style>
 body{margin:0;background:#0e1117;color:#cdd9e5;font:13px system-ui,Segoe UI,Arial}
 #bar{padding:8px 12px;background:#161b22;border-bottom:1px solid #30363d}
 button{background:#21262d;color:#cdd9e5;border:1px solid #30363d;border-radius:6px;
   padding:6px 12px;margin-right:6px;cursor:pointer}
 button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 #tip{position:fixed;pointer-events:none;background:#161b22cc;border:1px solid #30363d;
   border-radius:6px;padding:4px 8px;font-size:12px;display:none}
 #hint{color:#768390}
 canvas{display:block;cursor:grab}
</style></head><body>
<div id="bar">
  <span id="btns"></span>
  <span id="hint">drag = pan · wheel = zoom · hover a neuron to trace its links ·
   <span style="color:#58a6ff">blue +</span> / <span style="color:#f85149">red −</span> weight</span>
</div>
<canvas id="c"></canvas><div id="tip"></div>
<script>
const DATA = __DATA__;
let cur = Object.keys(DATA)[0];
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
let view = {x:60, y:0, s:1}, drag=null, hover=null, nodes=[];
function resize(){ cv.width=innerWidth; cv.height=innerHeight-42; layout(); draw(); }
addEventListener('resize', resize);

function layout(){
  const g = DATA[cur]; nodes=[];
  const L = g.layers.length;
  const colGap = (cv.width-160) / (L-1);
  for(let li=0; li<L; li++){
    const lay=g.layers[li], k=lay.shown;
    const x = 80 + li*colGap;
    const span = Math.min(cv.height-120, k*26);
    const y0 = (cv.height-span)/2;
    for(let i=0;i<k;i++){
      const y = k>1 ? y0 + i*(span/(k-1)) : cv.height/2;
      nodes.push({li, i, x, y, label:lay.label, size:lay.size, shown:k});
    }
  }
}
function nodeAt(li,i){ return nodes.find(n=>n.li===li&&n.i===i); }
function tx(x){ return x*view.s + view.x; }
function ty(y){ return y*view.s + view.y; }

function draw(){
  const g=DATA[cur];
  ctx.clearRect(0,0,cv.width,cv.height);
  // edges
  for(let bi=0; bi<g.blocks.length; bi++){
    const blk=g.blocks[bi];
    for(let o=0;o<blk.length;o++) for(let inp=0;inp<blk[o].length;inp++){
      const w=blk[o][inp], a=Math.min(1, Math.abs(w)/g.maxw);
      if(a<0.04) continue;
      const A=nodeAt(bi,inp), B=nodeAt(bi+1,o);
      const hot = hover && ((hover.li===bi&&hover.i===inp)||(hover.li===bi+1&&hover.i===o));
      ctx.strokeStyle = (w>=0? (hot?'rgba(88,166,255,':'rgba(56,110,190,')
                              : (hot?'rgba(248,81,73,':'rgba(190,70,66,')) + (hot? Math.min(1,a+0.3): a*0.5) + ')';
      ctx.lineWidth = (hot? 1.6:0.7)*Math.max(0.4,a);
      ctx.beginPath(); ctx.moveTo(tx(A.x),ty(A.y)); ctx.lineTo(tx(B.x),ty(B.y)); ctx.stroke();
    }
  }
  // nodes
  for(const n of nodes){
    const hot = hover && hover.li===n.li && hover.i===n.i;
    ctx.beginPath(); ctx.arc(tx(n.x),ty(n.y), (hot?6:4)*Math.max(0.6,view.s), 0, 7);
    ctx.fillStyle = hot? '#f0f6fc' : '#7d8da1'; ctx.fill();
  }
  // layer labels
  ctx.fillStyle='#cdd9e5'; ctx.textAlign='center'; ctx.font='12px system-ui';
  const seen=new Set();
  for(const n of nodes){ if(seen.has(n.li))continue; seen.add(n.li);
    const lines=n.label.split('\\n');
    lines.forEach((t,j)=> ctx.fillText(t, tx(n.x), 28 + j*14));
    if(n.size>n.shown) ctx.fillText('('+n.shown+' of '+n.size+')', tx(n.x), 28+lines.length*14);
  }
}

cv.addEventListener('mousedown',e=>{drag={x:e.clientX-view.x,y:e.clientY-view.y};cv.style.cursor='grabbing';});
addEventListener('mouseup',()=>{drag=null;cv.style.cursor='grab';});
cv.addEventListener('mousemove',e=>{
  if(drag){ view.x=e.clientX-drag.x; view.y=e.clientY-drag.y; draw(); return; }
  const mx=e.clientX, my=e.clientY-42; let best=null,bd=14;
  for(const n of nodes){ const d=Math.hypot(tx(n.x)-mx, ty(n.y)-my); if(d<bd){bd=d;best=n;} }
  const tip=document.getElementById('tip');
  if(best!==hover){ hover=best; draw(); }
  if(best){ tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
    tip.textContent = best.label.replace(/\\n/g,' ')+' · neuron #'+best.i; }
  else tip.style.display='none';
});
cv.addEventListener('wheel',e=>{ e.preventDefault();
  const f=e.deltaY<0?1.1:0.9, mx=e.clientX, my=e.clientY-42;
  view.x=mx-(mx-view.x)*f; view.y=my-(my-view.y)*f; view.s*=f; draw();
},{passive:false});

function setModel(m){ cur=m; view={x:60,y:0,s:1};
  document.querySelectorAll('#btns button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  layout(); draw(); }
const btns=document.getElementById('btns');
Object.keys(DATA).forEach(m=>{ const b=document.createElement('button');
  b.textContent='model '+m; b.dataset.m=m; b.onclick=()=>setModel(m); btns.appendChild(b); });
setModel(cur); resize();
</script></body></html>"""

with open("neuron_viz.html", "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(specs)))
print("wrote neuron_viz.html  (open in a browser)")


# ── reconstruct plain modules from the state_dict and export ONNX for Netron ─────
def shp(sd, k):
    return tuple(sd[k].shape)


def load_linear(lin, sd, wkey, bkey):
    lin.weight.data = sd[wkey].clone(); lin.bias.data = sd[bkey].clone()


def export_v1(sd):
    net = nn.Sequential(
        nn.Linear(141, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 12))
    load_linear(net[0], sd, "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias")
    load_linear(net[2], sd, "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias")
    load_linear(net[4], sd, "action_net.weight", "action_net.bias")
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 141), "v1_policy.onnx",
                      input_names=["obs_141"], output_names=["action_logits_12"], opset_version=17)
    return "v1_policy.onnx"


class V2Net(nn.Module):
    def __init__(self, sd):
        super().__init__()
        gch = shp(sd, "features_extractor.cnn.0.weight")[1]
        flat = shp(sd, "features_extractor.cnn_head.0.weight")[1]
        cnn_out = shp(sd, "features_extractor.cnn_head.0.weight")[0]
        vec_in = shp(sd, "features_extractor.vec_head.0.weight")[1]
        vec_out = shp(sd, "features_extractor.vec_head.0.weight")[0]
        self.cnn = nn.Sequential(nn.Conv2d(gch, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.cnn_head = nn.Sequential(nn.Linear(flat, cnn_out), nn.ReLU())
        self.vec_head = nn.Sequential(nn.Linear(vec_in, vec_out), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(cnn_out + vec_out, 512), nn.Tanh(),
                                   nn.Linear(512, 768), nn.Tanh(),
                                   nn.Linear(768, 512), nn.Tanh())
        self.action = nn.Linear(512, 12)
        self.gch = gch; self.vec_in = vec_in
        load_linear(self.cnn[0], sd, "features_extractor.cnn.0.weight", "features_extractor.cnn.0.bias")
        load_linear(self.cnn[2], sd, "features_extractor.cnn.2.weight", "features_extractor.cnn.2.bias")
        load_linear(self.cnn_head[0], sd, "features_extractor.cnn_head.0.weight", "features_extractor.cnn_head.0.bias")
        load_linear(self.vec_head[0], sd, "features_extractor.vec_head.0.weight", "features_extractor.vec_head.0.bias")
        load_linear(self.trunk[0], sd, "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias")
        load_linear(self.trunk[2], sd, "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias")
        load_linear(self.trunk[4], sd, "mlp_extractor.policy_net.4.weight", "mlp_extractor.policy_net.4.bias")
        load_linear(self.action, sd, "action_net.weight", "action_net.bias")

    def forward(self, grid, vec):
        f = torch.cat([self.cnn_head(self.cnn(grid)), self.vec_head(vec)], dim=1)
        return self.action(self.trunk(f))


def export_v2(sd):
    m = V2Net(sd).eval()
    g = torch.zeros(1, m.gch, 9, 13); v = torch.zeros(1, m.vec_in)
    torch.onnx.export(m, (g, v), "v2_policy.onnx",
                      input_names=["grid", "vector"], output_names=["action_logits_12"], opset_version=17)
    return "v2_policy.onnx"


for kind, fn in [("v1", export_v1), ("v2", export_v2)]:
    if kind not in SD:
        continue
    try:
        out = fn(SD[kind]); print("exported", out)
    except Exception as e:
        print(f"ONNX export {kind} FAILED: {e}")
