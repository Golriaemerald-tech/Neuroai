import fs from "node:fs";
import path from "node:path";
import * as model from "./model";
import * as memory from "./memory";
import { config } from "./config";
import { tokenize, learnVocabulary } from "./tokenizer";

function loadKnowledge():any[]{try{return JSON.parse(fs.readFileSync(config.paths.knowledge,"utf8"))||[]}catch{return[]}}
export function addKnowledge(input:any){const a=loadKnowledge();const e={id:`know_${Date.now()}_${Math.random().toString(16).slice(2)}`,question:String(input.question||""),answer:String(input.answer||""),category:input.category||"General",tags:Array.isArray(input.tags)?input.tags:[],createdAt:new Date().toISOString()};a.unshift(e);fs.writeFileSync(config.paths.knowledge,JSON.stringify(a),"utf8");learnVocabulary(`${e.question} ${e.answer}`);return e}
function overlap(a:string,b:string){const A=new Set(tokenize(a)),B=new Set(tokenize(b));if(!A.size||!B.size)return 0;let n=0;for(const t of A)if(B.has(t))n++;return n/Math.max(A.size,B.size)}
export function findBestKnowledge(q:string){const a=loadKnowledge();if(!a.length)return null;const w=model.loadWeights(),qe=model.embedding(w,q);let best:any=null,bs=0;for(const e of a){const o=overlap(q,e.question);let s=o;if(o>0)s=o*.8+Math.max(0,model.cosine(qe,model.embedding(w,e.question)))*.2;if(s>bs){bs=s;best=e}}return bs>=.25?{entry:best,score:bs}:null}
const greetings=new Set(["hi","hello","hey","yo","sup","hiya"]);
export function reply(message:string){const q=String(message||"").trim();if(!q)return{text:"I didn't catch that. Try asking me something.",matched:false};if(greetings.has(q.toLowerCase().replace(/[^a-z]/g,"")))return{text:"Hey! I'm NeuroAI. I'm running on your Node.js brain and learning from your local data.",matched:false};const m=findBestKnowledge(q),mem=memory.relevantContext(q,2);if(m){let t=m.entry.answer;if(mem.length)t+=`\n\nMemory recall: ${mem[0].content}`;return{text:t,matched:true,confidence:Number(m.score.toFixed(2)),source:m.entry}}if(mem.length)return{text:`I don't have a strong taught answer yet, but I remember: "${mem[0].content}"`,matched:false,recalledMemory:true};const w=model.loadWeights(),g=model.generate(w,q,18,.75);return{text:g||"I haven't learned that yet. Teach me and train the model.",matched:false,generatedByModel:!!g}}
export function load(){return loadKnowledge()}
export function parameterCount(w:any){return (w.E?.length||0)*(w.E?.[0]?.length||0)+(w.W1?.length||0)*(w.W1?.[0]?.length||0)+(w.W2?.length||0)*(w.W2?.[0]?.length||0)+(w.b1?.length||0)+(w.b2?.length||0)}
