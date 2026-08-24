import fs from "node:fs";
import path from "node:path";
import { config } from "./config";
import { loadVocab, tokenize } from "./tokenizer";

const C = config.model;
const rand = (s:number) => (Math.random()*2-1)*s;
const vec = (n:number, v=0) => new Array(n).fill(v);
const mat = (r:number,c:number,s:number) => Array.from({length:r},()=>Array.from({length:c},()=>rand(s)));

function softmax(x:number[]) {
  const m=Math.max(...x), e=x.map(v=>Math.exp(Math.max(-50,Math.min(50,v-m))));
  const s=e.reduce((a,b)=>a+b,0)||1; return e.map(v=>v/s);
}
export function freshWeights() {
  return {meta:{architecture:"embedding + context pooling + tanh + softmax",vocabSize:C.maxVocab,embeddingSize:C.embeddingSize,hiddenSize:C.hiddenSize,contextSize:C.contextSize,epochsTrained:0,stepsTrained:0,tokensTrained:0,loss:Math.log(C.maxVocab),learningRate:config.training.learningRate,parameters:C.maxVocab*C.embeddingSize+C.embeddingSize*C.hiddenSize+C.hiddenSize*C.maxVocab+C.hiddenSize+C.maxVocab,createdAt:new Date().toISOString(),updatedAt:new Date().toISOString()},E:mat(C.maxVocab,C.embeddingSize,Math.sqrt(1/C.embeddingSize)),W1:mat(C.embeddingSize,C.hiddenSize,Math.sqrt(1/C.embeddingSize)),b1:vec(C.hiddenSize),W2:mat(C.hiddenSize,C.maxVocab,Math.sqrt(1/C.hiddenSize)),b2:vec(C.maxVocab)};
}
export function saveWeights(w:any){fs.mkdirSync(path.dirname(config.paths.weights),{recursive:true});w.meta.updatedAt=new Date().toISOString();fs.writeFileSync(config.paths.weights,JSON.stringify(w),"utf8");}
export function loadWeights(){try{const w=JSON.parse(fs.readFileSync(config.paths.weights,"utf8"));if(w?.E&&w?.W1&&w?.W2)return w;}catch{}const w=freshWeights();saveWeights(w);return w;}
export function ids(text:string){const v=loadVocab();return tokenize(text).map(t=>v.word2idx[t]??0).filter(i=>i<C.maxVocab);}
export function forward(w:any, context:number[]){
  const ctx=context.length?context:[0], avg=vec(C.embeddingSize);
  for(const id of ctx)for(let j=0;j<C.embeddingSize;j++)avg[j]+=(w.E[id]?.[j]??0);
  for(let j=0;j<C.embeddingSize;j++)avg[j]/=ctx.length;
  const h=vec(C.hiddenSize);
  for(let k=0;k<C.hiddenSize;k++){let s=w.b1[k];for(let j=0;j<C.embeddingSize;j++)s+=avg[j]*w.W1[j][k];h[k]=Math.tanh(s);}
  const logits=vec(C.maxVocab);
  for(let o=0;o<C.maxVocab;o++){let s=w.b2[o];for(let k=0;k<C.hiddenSize;k++)s+=h[k]*w.W2[k][o];logits[o]=s;}
  return {avg,h,p:softmax(logits)};
}
export function trainStep(w:any,ctx:number[],target:number,lr:number){
  const f=forward(w,ctx), p=Math.max(1e-9,f.p[target]||1e-9), loss=-Math.log(p), dl=f.p.slice();dl[target]-=1;const dh=vec(C.hiddenSize);
  for(let k=0;k<C.hiddenSize;k++){let s=0;for(let o=0;o<C.maxVocab;o++){s+=dl[o]*w.W2[k][o];w.W2[k][o]-=lr*f.h[k]*dl[o];}dh[k]=s*(1-f.h[k]*f.h[k]);}
  for(let o=0;o<C.maxVocab;o++)w.b2[o]-=lr*dl[o];
  const da=vec(C.embeddingSize);
  for(let j=0;j<C.embeddingSize;j++){let s=0;for(let k=0;k<C.hiddenSize;k++){s+=w.W1[j][k]*dh[k];w.W1[j][k]-=lr*f.avg[j]*dh[k];}da[j]=s/ctx.length;}
  for(let k=0;k<C.hiddenSize;k++)w.b1[k]-=lr*dh[k];
  for(const id of ctx)for(let j=0;j<C.embeddingSize;j++)if(w.E[id])w.E[id][j]-=lr*da[j];
  return loss;
}
export function trainText(w:any,text:string,lr:number,maxSteps=config.training.stepsPerExample){
  const a=ids(text);if(a.length<2)return{loss:0,steps:0};let loss=0,steps=0;const stride=Math.max(1,Math.ceil((a.length-1)/maxSteps));
  for(let i=1;i<a.length;i+=stride){loss+=trainStep(w,a.slice(Math.max(0,i-C.contextSize),i),a[i],lr);steps++;}
  return{loss:loss/Math.max(1,steps),steps};
}
export function embedding(w:any,text:string){return forward(w,ids(text).slice(-C.contextSize)).h;}
export function cosine(a:number[],b:number[]){let d=0,aa=0,bb=0;for(let i=0;i<Math.min(a.length,b.length);i++){d+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return aa&&bb?d/(Math.sqrt(aa)*Math.sqrt(bb)):0;}
export function generate(w:any,prompt:string,maxTokens=24,temp=C.temperature){
  const a=ids(prompt),out:string[]=[],v=loadVocab();
  for(let n=0;n<maxTokens;n++){const f=forward(w,a.slice(-C.contextSize)),logits=f.p.map(p=>Math.log(Math.max(1e-9,p))/Math.max(.1,temp)),p=softmax(logits),top=p.map((x,id)=>({x,id})).sort((x,y)=>y.x-x.x).slice(0,C.topK);let r=Math.random()*top.reduce((s,x)=>s+x.x,0),id=top.at(-1)?.id??0;for(const x of top){r-=x.x;if(r<=0){id=x.id;break;}}if(id===1)break;a.push(id);const word=v.idx2word[String(id)];if(word&&word!=="<unk>")out.push(word);}
  return out.join(" ").replace(/\s+([!?.,;:])/g,"$1");
}
