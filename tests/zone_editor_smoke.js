const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, '..', 'tukuru', 'static', 'index.html'), 'utf8');
const block = 'const ZONE_COLORS' + html.split('const ZONE_COLORS')[1].split('function formData()')[0];

class Element {
  constructor(tag = 'div') { this.tag = tag; this.attrs = {}; this.children = []; this.style = {}; this.listeners = {}; this.hidden = false; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  getAttribute(key) { return this.attrs[key] ?? null; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(key, listener) { this.listeners[key] = listener; }
  getBoundingClientRect() { return {left:0, top:0, width:100, height:100}; }
  setPointerCapture(id) { this.capture = id; }
  hasPointerCapture(id) { return this.capture === id; }
  releasePointerCapture(id) { if (this.capture === id) this.capture = null; }
}
const elements = Object.fromEntries(['zonePreview','zoneEditorPanel','zoneItems','zoneNotes','zoneHelp','zoneKindLabel','deleteZoneBtn','applyZonesBtn','zoneKind'].map(id => [id,new Element()]));
elements.zoneKind.value = 'estanteria';
const cameraForm = {elements:{zones:{value:''}}};
const context = vm.createContext({
  document:{createElementNS:(ns,tag)=>new Element(tag),createTextNode:text=>({textContent:text})},
  $:id=>elements[id], el:tag=>new Element(tag), form:()=>cameraForm,
  say:()=>{},
});
vm.runInContext(block, context);
vm.runInContext('showZoneEditor("data:image/jpeg;base64,x", {}, "manual", "local")', context);
const svg = elements.zonePreview.children[1];
function event(type,target,x,y) { svg.listeners[type]({button:0,target,clientX:x,clientY:y,pointerId:1,preventDefault(){}}); }
const background = svg.children.find(child=>child.tag==='rect');
assert(background && background.style.pointerEvents==='all');
event('pointerdown',background,10,20);
event('pointermove',svg,50,70);
event('pointerup',svg,50,70);
let zones = JSON.parse(cameraForm.elements.zones.value);
assert.equal(zones.estanteria.length,1);
assert.deepEqual(zones.estanteria[0][0],[0.1,0.2]);
assert.deepEqual(zones.estanteria[0][2],[0.5,0.7]);
let polygon = svg.children.find(child=>child.tag==='polygon');
event('pointerdown',polygon,20,30);
event('pointermove',svg,30,30);
event('pointerup',svg,30,30);
zones = JSON.parse(cameraForm.elements.zones.value);
assert(Math.abs(zones.estanteria[0][0][0]-0.2)<1e-9);
let handle = svg.children.find(child=>child.tag==='circle' && child.attrs['data-point']==='0');
event('pointerdown',handle,20,20);
event('pointermove',svg,15,15);
event('pointerup',svg,15,15);
zones = JSON.parse(cameraForm.elements.zones.value);
assert.deepEqual(zones.estanteria[0][0],[0.15,0.15]);
polygon = svg.children.find(child=>child.tag==='polygon');
event('pointerdown',polygon,30,30);
event('pointermove',svg,200,200);
event('pointerup',svg,200,200);
zones = JSON.parse(cameraForm.elements.zones.value);
assert(zones.estanteria[0].every(([x,y])=>x>=0 && x<=1 && y>=0 && y<=1));
elements.zoneItems.children[0].onclick();
assert.equal(elements.deleteZoneBtn.disabled,false);
assert.equal(elements.deleteZoneBtn.hidden,false);
vm.runInContext('deleteSelectedZone()', context);
zones = JSON.parse(cameraForm.elements.zones.value);
assert.equal(zones.estanteria,undefined);
assert.equal(elements.deleteZoneBtn.disabled,true);
console.log('draw, move, resize, select, delete: OK');
