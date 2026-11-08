import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { MODEL, messages } from './routes.mjs';
env.allowLocalModels = false;
let generator, device;
const hash = async text => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))), b=>b.toString(16).padStart(2,'0')).join('');
self.onmessage = async ({data}) => {
  try {
    if(data.type === 'load') {
      device = data.device === 'webgpu' ? 'webgpu' : 'wasm';
      generator = await pipeline('text-generation',MODEL.id,{
        revision:MODEL.revision, dtype:MODEL.dtype, device,
        progress_callback:p=>self.postMessage({type:'progress',text:p.status,progress:p.progress})
      });
      self.postMessage({type:'ready'});
    } else if(data.type === 'generate') {
      if(!generator) throw Error('Load the model first.');
      const started = performance.now();
      const result = await generator(messages(data.prompt,data.profile),{
        max_new_tokens:192, do_sample:false, return_full_text:false
      });
      const generated = result[0]?.generated_text;
      const text = typeof generated === 'string' ? generated : generated?.at(-1)?.content;
      if(typeof text !== 'string' || !text.trim()) throw Error('The model returned no answer.');
      self.postMessage({type:'answer',text,receipt:{
        schema:'meok.browser-receipt/0.1', id:crypto.randomUUID(), observed_at:new Date().toISOString(),
        status:'RUNTIME_OBSERVED',quality:'UNMEASURED',model:MODEL,device,runtime:'Transformers.js 3.8.1 / ONNX Runtime',
        specialist:data.profile,prompt_sha256:await hash(data.prompt),output_sha256:await hash(text),
        generation:{max_new_tokens:192,do_sample:false},duration_ms:Math.round(performance.now()-started),
        signature:null,limitations:['Browser-generated receipt, unsigned.','Prompt profile, not trained specialist.','Answer correctness not independently evaluated.']
      }});
    }
  } catch(error) { self.postMessage({type:'error',text:String(error.message || 'Model execution failed')}); }
};
