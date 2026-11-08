export const MODEL = Object.freeze({
  id: 'HuggingFaceTB/SmolLM2-360M-Instruct',
  revision: 'a10cc1512eabd3dde888204e902eca88bddb4951', dtype: 'q4'
});
export const PROFILES = Object.freeze({
  general: 'Help with the request directly and concisely.',
  evidence: 'Separate observations, unknowns and proposed checks. Never invent sources or signed receipts.',
  code: 'Explain code clearly. Do not claim code ran unless an execution result was supplied.'
});
export function messages(prompt, profile) {
  if (typeof prompt !== 'string' || !prompt.trim() || prompt.length > 2000) throw Error('Use 1–2000 characters.');
  if (!Object.hasOwn(PROFILES, profile)) throw Error('Unknown specialist.');
  return [{role:'system', content:'You are MEOK, a bounded assistant. State uncertainty. You cannot execute tools, approve changes or issue legal verdicts. ' + PROFILES[profile]}, {role:'user', content:prompt}];
}
