import fs from "node:fs";
import * as model from "./model";
import { config } from "./config";
import { load } from "./brain";

let running=false, timer:NodeJS.Timeout|null=null;
export const state:any={status:"idle",epoch:0,steps:0,loss:0,learningRate:config.training.learningRate,lossHistory:[]};
export function trainEpoch(){const w=model.loadWeights(), entries=load();if(!entries.length)return state;const lr=Math.max(.001,config.training.learningRate*Math.pow(.98,w.meta.epochsTrained||0));let loss=0,steps=0;for(const e of entries.slice(0,500)){const r=model.trainText(w,`${e.question} ${e.answer}`,lr);loss+=r.loss;steps+=r.steps;}w.meta.epochsTrained=(w.meta.epochsTrained||0)+1;w.meta.stepsTrained=(w.meta.stepsTrained||0)+steps;w.meta.tokensTrained=(w.meta.tokensTrained||0)+steps;w.meta.loss=loss/Math.max(1,entries.slice(0,500).length);w.meta.learningRate=lr;model.saveWeights(w);state.epoch=w.meta.epochsTrained;state.steps=w.meta.stepsTrained;state.loss=w.meta.loss;state.learningRate=lr;state.lossHistory.push(state.loss);state.lossHistory=state.lossHistory.slice(-100);return state}
export function start(){if(running)return state;running=true;state.status="running";const loop=()=>{if(!running)return;try{trainEpoch()}catch(e){state.status="error";console.error(e)}timer=setTimeout(loop,100)};loop();return state}
export function stop(){running=false;if(timer)clearTimeout(timer);timer=null;state.status="paused";return state}
export function getState(){const w=model.loadWeights();return{...state,status:running?"running":state.status,epoch:w.meta.epochsTrained||0,steps:w.meta.stepsTrained||0,loss:w.meta.loss,learningRate:w.meta.learningRate}}
