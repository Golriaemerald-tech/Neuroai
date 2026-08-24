import fs from "node:fs";
import { config } from "./config";
import { tokenize } from "./tokenizer";

function loadAll():any[]{try{return JSON.parse(fs.readFileSync(config.paths.memory,"utf8"))||[]}catch{return[]}}
function saveAll(a:any[]){fs.mkdirSync(require("node:path").dirname(config.paths.memory),{recursive:true});fs.writeFileSync(config.paths.memory,JSON.stringify(a,null,2),"utf8")}
export function addMemory(content:string,source="conversation"){const a=loadAll();const item={id:`mem_${Date.now()}_${Math.random().toString(16).slice(2)}`,content,source,createdAt:new Date().toISOString()};a.unshift(item);saveAll(a);return item}
export function searchMemories(q:string){const qt=new Set(tokenize(q));return loadAll().map(e=>{const et=new Set(tokenize(e.content));let n=0;for(const t of qt)if(et.has(t))n++;return{...e,score:n/Math.max(1,qt.size)}}).filter(e=>e.score>0).sort((a,b)=>b.score-a.score)}
export function relevantContext(q:string,n=3){return searchMemories(q).slice(0,n)}
export {loadAll};
