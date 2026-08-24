import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { URL } from "node:url";
import { config } from "./config";
import * as brain from "./brain";
import * as memory from "./memory";
import * as model from "./model";
import * as tokenizer from "./tokenizer";
import * as trainer from "./trainer";

const PORT=Number(process.env.PORT||3000);
const DIST=path.join(__dirname,"dist");
function send(res:http.ServerResponse,status:number,data:any,type="application/json"){res.statusCode=status;res.setHeader("Content-Type",type);res.setHeader("Cache-Control","no-store");res.end(type.startsWith("application/json")?JSON.stringify(data):data)}
function body(req:http.IncomingMessage):Promise<any>{return new Promise((resolve,reject)=>{let raw="";req.on("data",c=>{raw+=c;if(raw.length>2e6)reject(new Error("Request too large"))});req.on("end",()=>{try{resolve(raw?JSON.parse(raw):{})}catch{reject(new Error("Invalid JSON"))}});req.on("error",reject)})}
function stats(){const w=model.loadWeights(),k=brain.load(),m=memory.loadAll();return{knowledgeEntries:k.length,memoryEntries:m.length,vocabSize:tokenizer.vocabSize(),parameters:brain.parameterCount(w),epochs:w.meta.epochsTrained||0,steps:w.meta.stepsTrained||0,loss:w.meta.loss,learningRate:w.meta.learningRate,training:trainer.getState(),architecture:w.meta.architecture}}
async function api(req:http.IncomingMessage,res:http.ServerResponse,u:URL){
 const method=req.method||"GET", p=u.pathname.replace(/\/+$/,"")||"/";
 if(method==="POST"&&p==="/api/chat"){const b=await body(req);return send(res,200,brain.reply(String(b.message||"")))}
 if(method==="GET"&&p==="/api/stats")return send(res,200,stats());
 if(method==="GET"&&p==="/api/training")return send(res,200,trainer.getState());
 if(method==="POST"&&p==="/api/training/start"){trainer.start();return send(res,200,trainer.getState())}
 if(method==="POST"&&p==="/api/training/stop"){trainer.stop();return send(res,200,trainer.getState())}
 if(method==="GET"&&p==="/api/model")return send(res,200,model.loadWeights().meta);
 if(method==="GET"&&p==="/api/knowledge"){let k=brain.load();const q=u.searchParams.get("q")?.toLowerCase();if(q)k=k.filter((x:any)=>`${x.question} ${x.answer}`.toLowerCase().includes(q));return send(res,200,k.slice(0,1000))}
 if(method==="POST"&&p==="/api/knowledge"){const b=await body(req);return send(res,201,brain.addKnowledge(b))}
 if(method==="GET"&&p==="/api/memory")return send(res,200,memory.loadAll().slice(0,1000))
 if(method==="POST"&&p==="/api/memory"){const b=await body(req);return send(res,201,memory.addMemory(String(b.content||""),String(b.source||"manual")))}
 return send(res,404,{error:"API route not found"});
}
function staticFile(res:http.ServerResponse,u:URL){let p=u.pathname==="/"?"index.html":u.pathname.slice(1);p=path.normalize(p);if(p.startsWith(".."))return send(res,403,{error:"Forbidden"});const file=path.join(DIST,p);if(!fs.existsSync(file)||!fs.statSync(file).isFile())return send(res,404,{error:"Not found"});const ext=path.extname(file);const types:any={".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8",".svg":"image/svg+xml",".json":"application/json"};send(res,200,fs.readFileSync(file),types[ext]||"application/octet-stream")}
const server=http.createServer(async(req,res)=>{try{const u=new URL(req.url||"/","http://localhost");if(u.pathname.startsWith("/api/"))await api(req,res,u);else staticFile(res,u)}catch(e:any){console.error(e);send(res,500,{error:e?.message||"Internal server error"})}});
server.listen(PORT,"0.0.0.0",()=>{console.log(`\n🧠 NeuroAI Node.js + TSX\n   http://localhost:${PORT}\n   Native Node HTTP server — no Express\n   Auto-training: ${config.training.auto?"enabled":"disabled"}\n`)});
if(config.training.auto)trainer.start();
