# ruff: noqa: E501  -- HTML/CSS/JS template strings, not logic
"""The forecast explorer section of `reports/DEMO.html` (task P5): a buyer browsing the forecast.

One self-contained block: the top styles are embedded as JSON (`explorer_data`), the page filters,
sorts and shows a detail panel (weekly trajectory, SHAP drivers, guards, the generated concept when
one exists) with vanilla JS only: no server, no CDN, no fetch. All text from the data is inserted
with `textContent`, never as HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path("reports/tables/explorer_styles.json")

FEATURE_WORDS = {
    "lag_1": "last week's sales level for this style",
    "lag_2": "sales two weeks ago",
    "lag_52": "the same week last year",
    "slope_13w": "the 13-week sales trend",
    "n_active_articles_level": "how many products of this style are on sale",
    "n_active_articles_trend_13w": "whether the range is growing or shrinking",
    "fourier_sin_1": "the time-of-year pattern",
    "fourier_cos_1": "the time-of-year pattern",
    "garment_group_name": "the kind of garment",
    "perceived_colour_master_name": "the colour",
    "product_type_name": "the product type",
    "graphical_appearance_name": "the pattern",
    "index_group_name": "the department",
    "share_garment_group": "its share of its garment group's sales",
    "share_index_group": "its share of its department's sales",
    "price_index_level": "how discounted it is",
}


def payload(concept_images: dict[str, str]) -> str:
    """The JSON blob: styles, plain-language feature names, and concept images keyed by style."""
    styles = json.loads(DATA.read_text(encoding="utf-8"))
    blob = {"styles": styles, "words": FEATURE_WORDS, "concepts": concept_images}
    return json.dumps(blob, separators=(",", ":")).replace("</", "<\\/")


def section(concept_images: dict[str, str]) -> str:
    """The explorer HTML (controls, table, detail panel) with its data and script embedded."""
    return f"""<section id=explorer><h2>Browse the forecast</h2>
<p class=lede>The 200 styles the model expects to sell hardest per product over the 13 weeks after 21 September 2020 (all three sanity guards passed). Filter to your own category, sort by predicted intensity or by growth against recent sales, and open a row for its weekly history, the reasons behind its forecast and, where one exists, the picture generated from it.</p>
<div class=xp>
<div class=xp-main>
<form class=xp-controls onsubmit="return false" aria-label="Filters">
<label class=xp-field><span>Search</span><input type=search id=xq placeholder="e.g. knit, red, dress" autocomplete=off></label>
<label class=xp-field><span>Peak season</span><select id=xs><option value="">Any season</option><option>spring</option><option>summer</option><option>autumn</option><option>winter</option></select></label>
<label class=xp-field><span>Product type</span><select id=xt><option value="">All product types</option></select></label>
<label class=xp-field><span>Sort by</span><select id=xo><option value=pred>Predicted intensity</option><option value=growth>Growth against recent sales</option><option value=rank>Rank</option></select></label>
<label class=xp-check><input type=checkbox id=xg> Rising only (forecast at least 10% above the last 13 weeks)</label>
<fieldset class=xp-colours><legend>Colour</legend><div id=xc class=xp-chips></div></fieldset>
<button type=button id=xr class=xp-reset>Reset filters</button>
</form>
<p id=xn class=xp-count aria-live=polite></p>
<div class=scroll><table class=xp-table><thead><tr><th scope=col>#</th><th scope=col>Style</th><th scope=col class=n>Predicted</th><th scope=col class=n>Growth</th><th scope=col>Peak season</th></tr></thead><tbody id=xb></tbody></table></div>
<div id=xe class=xp-empty hidden><p>No style matches these filters.</p><button type=button id=xe2 class=xp-reset>Reset filters</button></div>
<button type=button id=xm class=xp-more hidden>Show 25 more</button>
</div>
<aside id=xd class=xp-detail aria-live=polite aria-label="Style detail"><p class=xp-hint>Select a style to see its history, its drivers and its guards.</p></aside>
</div>
<p class=gloss>"Intensity" is units sold per product on sale per week, so a style is not ranked highly just for having many products. "Growth" is the forecast divided by the last 13 weeks. "Peak season" is the season with the highest historical mean. The list is the model's raw output, so it includes intimates and swimwear that the concept selection rule excludes. Numbers are the model's, read from <code>reports/tables/explorer_styles.json</code>.</p>
<script type="application/json" id=xdata>{payload(concept_images)}</script>
<script>{JS}</script>
</section>"""


CSS = """
.xp{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,2fr);gap:36px;align-items:start}
.xp-controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px 16px;align-items:end;margin:0 0 8px}
.xp-field{display:flex;flex-direction:column;gap:4px;font-size:.82rem;color:var(--muted)}
.xp-field input,.xp-field select{font:inherit;font-size:.95rem;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-bottom:2px solid var(--ink);padding:8px 8px;border-radius:0;min-width:0}
.xp-field input:focus-visible,.xp-field select:focus-visible,.xp-check input:focus-visible,.xp-reset:focus-visible,.xp-more:focus-visible,.xp-chip:focus-visible,.xp-table tbody tr:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.xp-check{grid-column:1/-1;font-size:.9rem;display:flex;gap:8px;align-items:center}
.xp-colours{grid-column:1/-1;border:0;padding:0;margin:0}.xp-colours legend{font-size:.82rem;color:var(--muted);padding:0;margin-bottom:6px}
.xp-chips{display:flex;flex-wrap:wrap;gap:6px}
.xp-chip{font:inherit;font-size:.85rem;padding:5px 11px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);cursor:pointer;border-radius:999px}
.xp-chip[aria-pressed=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.xp-reset,.xp-more{font:inherit;font-size:.9rem;padding:8px 16px;border:1px solid var(--ink);background:transparent;color:var(--ink);cursor:pointer;justify-self:start}
.xp-reset:hover,.xp-more:hover{background:var(--ink);color:var(--paper)}
.xp-count{margin:18px 0 6px;color:var(--muted);font-size:.9rem}
.xp-table{font-size:.93rem}.xp-table th{text-align:left;border-bottom:2px solid var(--ink);padding:8px 8px;font-size:.8rem;font-weight:600}
.xp-table td{padding:10px 8px;border-top:1px solid var(--rule);vertical-align:middle}
.xp-table .n{text-align:right;font-variant-numeric:tabular-nums}
.xp-table tbody tr{cursor:pointer}.xp-table tbody tr:hover{background:var(--wash)}
.xp-table tbody tr[aria-selected=true]{background:var(--wash);box-shadow:inset 3px 0 0 var(--accent)}
.xp-name{font-weight:600;display:block}.xp-sub{color:var(--muted);font-size:.8rem}
.xp-bar{display:inline-block;height:6px;background:var(--accent);vertical-align:middle;margin-right:8px;min-width:2px}
.xp-tag{display:inline-block;font-size:.7rem;padding:1px 7px;border:1px solid var(--accent);color:var(--accent);margin-left:6px;border-radius:999px;vertical-align:middle}
.xp-empty{padding:28px 0}.xp-more{margin:18px 0 0}
.xp-detail{position:sticky;top:14px;border-top:2px solid var(--ink);padding-top:14px;min-height:200px}
.xp-detail h3{font:400 1.5rem/1.2 var(--serif);margin:0 0 4px}.xp-hint{color:var(--muted)}
.xp-attrs{color:var(--muted);font-size:.86rem;margin:0 0 14px}
.xp-nums{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 16px}
.xp-nums div{border-top:1px solid var(--rule);padding-top:6px}.xp-nums b{display:block;font:400 1.5rem/1.1 var(--serif);font-variant-numeric:tabular-nums}.xp-nums span{font-size:.76rem;color:var(--muted)}
.xp-spark{width:100%;height:auto;display:block;margin:4px 0 2px}.xp-spark .l{fill:none;stroke:var(--ink);stroke-width:1.6}.xp-spark .f{stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 3}.xp-spark .a{stroke:var(--rule);stroke-width:1}.xp-spark text{font:11px var(--sans);fill:var(--muted)}
.xp-detail h4{margin:16px 0 6px}
.xp-shap{list-style:none;padding:0;margin:0;font-size:.85rem}.xp-shap li{display:grid;grid-template-columns:minmax(0,1fr) 88px;gap:8px;align-items:center;padding:3px 0}
.xp-sb{position:relative;height:8px;background:var(--wash)}.xp-sb i{position:absolute;top:0;height:8px}.xp-sb .p{background:var(--accent)}.xp-sb .m{background:var(--ink)}.xp-sb u{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.xp-guards{list-style:none;padding:0;margin:0;font-size:.86rem}.xp-guards li{padding:4px 0;border-top:1px solid var(--rule)}
.xp-concept{margin:16px 0 0}.xp-concept img{width:100%;display:block;background:var(--wash)}.xp-concept figcaption{font-size:.82rem;color:var(--muted);padding-top:6px}
@media(max-width:1000px){.xp{grid-template-columns:1fr}.xp-detail{position:static}.xp-controls{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(prefers-reduced-motion:no-preference){.xp-chip,.xp-reset,.xp-more{transition:background .15s,color .15s}}
"""

JS = r"""
(function(){
"use strict";
var el=function(i){return document.getElementById(i)};
var box=el("explorer");
var blob;
try{blob=JSON.parse(el("xdata").textContent)}catch(e){
  el("xn").textContent="The forecast data could not be read from this page.";return}
var S=blob.styles,W=blob.words,CI=blob.concepts;
var PAGE=25,shown=PAGE,sel=null,colours={};
var S_={q:"",season:"",type:"",sort:"pred",rising:false};
var seasonsTitle=function(s){return s.charAt(0).toUpperCase()+s.slice(1)};
var mk=function(tag,cls,txt){var n=document.createElement(tag);if(cls)n.className=cls;if(txt!=null)n.textContent=txt;return n};
var nice=function(s){return s.charAt(0).toUpperCase()+s.slice(1)};
function fill(){
  var types={},cols={},i;
  for(i=0;i<S.length;i++){types[S[i].product_type_name]=(types[S[i].product_type_name]||0)+1;cols[S[i].perceived_colour_master_name]=(cols[S[i].perceived_colour_master_name]||0)+1}
  var tsel=el("xt");
  Object.keys(types).sort().forEach(function(t){var o=mk("option",null,t+" ("+types[t]+")");o.value=t;tsel.appendChild(o)});
  var cbox=el("xc");
  Object.keys(cols).sort(function(a,b){return cols[b]-cols[a]}).forEach(function(c){
    var b=mk("button","xp-chip",c+" "+cols[c]);b.type="button";b.setAttribute("aria-pressed","false");b.dataset.c=c;
    b.addEventListener("click",function(){colours[c]=!colours[c];b.setAttribute("aria-pressed",colours[c]?"true":"false");shown=PAGE;render()});
    cbox.appendChild(b)});
}
function match(s){
  if(S_.season&&s.season!==S_.season)return false;
  if(S_.type&&s.product_type_name!==S_.type)return false;
  if(S_.rising&&!(s.growth!=null&&s.growth>=1.1))return false;
  var any=false,k;for(k in colours){if(colours[k]){any=true;break}}
  if(any&&!colours[s.perceived_colour_master_name])return false;
  if(S_.q){var hay=(s.key+" "+s.season).toLowerCase(),ts=S_.q.toLowerCase().split(/\s+/);
    for(var i=0;i<ts.length;i++){if(ts[i]&&hay.indexOf(ts[i])<0)return false}}
  return true}
function rows(){
  var r=S.filter(match);
  var f={pred:function(a,b){return b.pred-a.pred},growth:function(a,b){return (b.growth||0)-(a.growth||0)},rank:function(a,b){return a.rank-b.rank}}[S_.sort];
  return r.sort(f)}
var maxPred=0;S.forEach(function(s){if(s.pred>maxPred)maxPred=s.pred});
function render(){
  var r=rows(),tb=el("xb");tb.textContent="";
  var n=Math.min(shown,r.length);
  for(var i=0;i<n;i++){(function(s){
    var tr=mk("tr");tr.tabIndex=0;tr.setAttribute("role","button");tr.dataset.k=s.key;
    tr.setAttribute("aria-selected",sel===s.key?"true":"false");
    tr.appendChild(mk("td",null,String(s.rank)));
    var td=mk("td");var nm=mk("span","xp-name",s.perceived_colour_master_name+" "+s.product_type_name.toLowerCase());
    if(s.concept){nm.appendChild(mk("span","xp-tag","concept"))}
    td.appendChild(nm);td.appendChild(mk("span","xp-sub",s.graphical_appearance_name+" · "+s.garment_group_name+" · "+s.index_group_name));tr.appendChild(td);
    var tp=mk("td","n");var bar=mk("span","xp-bar");bar.style.width=Math.max(2,Math.round(60*s.pred/maxPred))+"px";tp.appendChild(bar);tp.appendChild(document.createTextNode(s.pred.toFixed(1)));tr.appendChild(tp);
    tr.appendChild(mk("td","n",s.growth!=null?s.growth.toFixed(2)+"×":"n/a"));
    tr.appendChild(mk("td",null,nice(s.season)));
    var open=function(){choose(s.key,true)};
    tr.addEventListener("click",open);
    tr.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();open()}});
    tb.appendChild(tr)})(r[i])}
  el("xn").textContent=r.length+" of "+S.length+" styles"+(r.length>n?" (showing "+n+")":"");
  el("xe").hidden=r.length!==0;
  el("xm").hidden=!(r.length>n);
  el("xm").textContent="Show "+Math.min(PAGE,r.length-n)+" more"}
function spark(s){
  var t=s.traj,n=t.length,w=420,h=120,pl=30,pb=18,pt=8,pr=8;
  if(!n)return mk("p","gloss","No weekly history in the panel.");
  var mx=Math.max.apply(null,t.concat([s.pred]))*1.08,mn=0;
  var X=function(i){return pl+(w-pl-pr)*i/Math.max(1,n-1)},Y=function(v){return pt+(h-pt-pb)*(1-(v-mn)/(mx-mn))};
  var svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.setAttribute("viewBox","0 0 "+w+" "+h);svg.setAttribute("class","xp-spark");svg.setAttribute("role","img");
  svg.setAttribute("aria-label","Weekly units per product on sale, last "+n+" weeks, with the forecast level");
  var ns="http://www.w3.org/2000/svg";
  function add(tag,attrs,txt){var e=document.createElementNS(ns,tag);for(var k in attrs)e.setAttribute(k,attrs[k]);if(txt!=null)e.textContent=txt;svg.appendChild(e);return e}
  add("line",{x1:pl,y1:Y(0),x2:w-pr,y2:Y(0),"class":"a"});
  add("text",{x:2,y:Y(0)+3},"0");add("text",{x:2,y:Y(mx/1.08)+3},Math.round(mx/1.08));
  var d="";for(var i=0;i<n;i++){d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(t[i]).toFixed(1)}
  add("path",{d:d,"class":"l"});
  add("line",{x1:X(n-14),y1:Y(s.pred),x2:w-pr,y2:Y(s.pred),"class":"f"});
  add("text",{x:w-pr,y:Y(s.pred)-4,"text-anchor":"end"},"forecast "+s.pred.toFixed(1));
  add("text",{x:pl,y:h-3},n+" weeks to 21 Sep 2020");
  return svg}
function shap(s){
  var ul=mk("ul","xp-shap"),m=0;s.shap.forEach(function(x){m=Math.max(m,Math.abs(x[1]))});
  s.shap.forEach(function(x){
    var li=mk("li"),lab=mk("span",null,nice(W[x[0]]||x[0]));li.appendChild(lab);
    var b=mk("span","xp-sb");b.appendChild(mk("u"));
    var i=mk("i",x[1]>=0?"p":"m");var wd=Math.round(50*Math.abs(x[1])/(m||1));i.style.width=wd+"%";i.style[x[1]>=0?"left":"right"]="50%";
    i.setAttribute("title",(x[1]>=0?"raises":"lowers")+" the forecast by "+Math.abs(x[1]).toFixed(3)+" (log scale)");
    b.appendChild(i);li.appendChild(b);ul.appendChild(li)});
  return ul}
function detail(s){
  var d=el("xd");d.textContent="";
  d.appendChild(mk("h3",null,s.perceived_colour_master_name+" "+s.product_type_name.toLowerCase()));
  d.appendChild(mk("p","xp-attrs",s.graphical_appearance_name+" · "+s.garment_group_name+" · "+s.index_group_name));
  var nums=mk("div","xp-nums");
  [[s.pred.toFixed(1),"predicted units per product per week"],["#"+s.rank+" of "+S.length,"rank by predicted intensity"],[s.growth!=null?s.growth.toFixed(2)+"×":"n/a","against the last 13 weeks ("+s.trail.toFixed(1)+")"]].forEach(function(p){
    var c=mk("div");c.appendChild(mk("b",null,p[0]));c.appendChild(mk("span",null,p[1]));nums.appendChild(c)});
  d.appendChild(nums);
  d.appendChild(mk("h4",null,"Weekly history"));d.appendChild(spark(s));
  d.appendChild(mk("h4",null,"Why the model expects this"));d.appendChild(shap(s));
  d.appendChild(mk("p","gloss","Bars to the right raise the forecast, to the left lower it (SHAP, per style)."));
  d.appendChild(mk("h4",null,"Guards passed"));
  var g=mk("ul","xp-guards"),gd=s.guard;
  [["Commercial scale: "+gd.n_active+" products on sale on average (need 10 or more)"],["Not heavily discounted: price index "+gd.price_index.toFixed(2)+" (need 0.85 or more)"],["Still selling: active "+gd.weeks_active+" of the last 52 weeks (need 26 or more)"]].forEach(function(t){g.appendChild(mk("li",null,t[0]))});
  d.appendChild(g);
  d.appendChild(mk("p","gloss","Historically strongest in "+s.season+" ("+s.season_index.toFixed(2)+"× its overall average)."));
  if(CI[s.key]){var f=mk("figure","xp-concept"),im=mk("img");im.src=CI[s.key];im.alt="Generated concept for this style";f.appendChild(im);f.appendChild(mk("figcaption",null,"The concept generated from this style (see the evidence section below)."));d.appendChild(f)}
}
function choose(k,focus){
  sel=k;var s=S.filter(function(x){return x.key===k})[0];if(!s)return;
  detail(s);
  Array.prototype.forEach.call(el("xb").children,function(tr){tr.setAttribute("aria-selected",tr.dataset.k===k?"true":"false")});
  if(focus&&window.matchMedia("(max-width:1000px)").matches){el("xd").scrollIntoView({behavior:"smooth",block:"start"})}}
function reset(){
  S_={q:"",season:"",type:"",sort:"pred",rising:false};colours={};shown=PAGE;
  el("xq").value="";el("xs").value="";el("xt").value="";el("xo").value="pred";el("xg").checked=false;
  Array.prototype.forEach.call(el("xc").children,function(b){b.setAttribute("aria-pressed","false")});render()}
fill();
el("xq").addEventListener("input",function(e){S_.q=e.target.value;shown=PAGE;render()});
el("xs").addEventListener("change",function(e){S_.season=e.target.value;shown=PAGE;render()});
el("xt").addEventListener("change",function(e){S_.type=e.target.value;shown=PAGE;render()});
el("xo").addEventListener("change",function(e){S_.sort=e.target.value;shown=PAGE;render()});
el("xg").addEventListener("change",function(e){S_.rising=e.target.checked;shown=PAGE;render()});
el("xr").addEventListener("click",reset);el("xe2").addEventListener("click",reset);
el("xm").addEventListener("click",function(){shown+=PAGE;render()});
box.addEventListener("keydown",function(e){if(e.key==="Escape"&&sel){sel=null;el("xd").textContent="";el("xd").appendChild(mk("p","xp-hint","Select a style to see its history, its drivers and its guards."));render()}});
render();
var first=S.filter(function(x){return x.concept})[0];if(first){choose(first.key,false)}
})();
"""
