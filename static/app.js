const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const resumeKinds=['项目','实习经历','学历','工作技能','工作经历','其他'];
const labels = {ai:'AI 应用开发', agent:'Agent 开发', fullstack:'全栈开发'};
const state = {track:'custom', mode:'demo', resume:null, session:null, busy:false, config:{},
  recorder:null, recognition:null, stream:null, chunks:[], blob:null, blobUrl:null, audio:null, speechToken:0};
const esc = s => String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const fmt = n => {n=Math.floor(n||0);return String(Math.floor(n/60)).padStart(2,'0')+':'+String(n%60).padStart(2,'0')};
function status(id, text, error=false) {const e=$(id);e.textContent=text;e.className=error?'status error':'status'}
async function api(url, body, method) {
  const res = await fetch(url, {method:method || (body ? 'POST':'GET'),
    headers:body?{'Content-Type':'application/json'}:{}, body:body?JSON.stringify(body):undefined});
  const data = await res.json();
  if(!res.ok) throw Error(typeof data.detail==='string'?data.detail:'参数无效，请检查填写内容');
  return data;
}
async function run(task, errorTarget='#audioStatus') {
  if(state.busy)return;
  state.busy=true;
  document.querySelectorAll('button').forEach(b=>{b.dataset.wasDisabled=b.disabled?'1':'0';b.disabled=true});
  try {await task()} catch(e) {status(errorTarget,e.message,true);$(errorTarget).scrollIntoView({behavior:'smooth',block:'center'})}
  finally {state.busy=false;document.querySelectorAll('button').forEach(b=>{b.disabled=b.dataset.wasDisabled==='1'});updateControls()}
}
function updateControls() {
  const s=state.session, recording=!!(state.recorder||state.recognition);
  $('#startBtn').disabled=!state.resume || (state.mode==='live'&&!state.config.llm);
  $('#liveMode').disabled=!state.config.llm;
  $('#startBtn').textContent=state.resume?'确认经历，开始面试':'请先解析简历';
  $('#startFromResume').disabled=$('#startBtn').disabled;
  $('#sendBtn').disabled=!s||s.status!=='active'||s.pending||recording;
  $('#recordBtn').disabled=!s||s.status!=='active'||s.pending;
  $('#answer').disabled=!s||s.status!=='active'||s.pending;
  $('#retryBtn').disabled=!s?.pending||s.status!=='active';
  $('#pauseBtn').disabled=!s||s.status==='ended'||recording;
  $('#endBtn').disabled=!s||s.status==='ended'||recording;
  $('#retryAudioBtn').disabled=!state.blob||!state.config.stt||recording;
  $('#deleteBtn').classList.toggle('hidden',!s);
  $('#deleteBtn').disabled=recording;
}
function stopSpeech() {
  state.speechToken++;
  window.speechSynthesis?.cancel();
  if(state.audio){state.audio.pause();URL.revokeObjectURL(state.audio.src);state.audio=null}
}
async function speak(text) {
  stopSpeech();
  if(!text)return;
  const token=state.speechToken;
  status('#audioStatus','准备朗读……');
  try {
    if(state.config.tts){
      const r=await fetch('/api/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.slice(0,2400)})});
      if(!r.ok)throw Error('云端语音播放失败，可以阅读文字或稍后重播');
      const blob=await r.blob();if(token!==state.speechToken)return;
      state.audio=new Audio(URL.createObjectURL(blob));
      state.audio.onended=()=>status('#audioStatus','朗读结束，可以回答。');
      await state.audio.play();
    }else{
      if(!window.speechSynthesis)throw Error('当前浏览器没有语音合成，请配置 TTS 服务');
      const voice=speechSynthesis.getVoices().find(v=>/^zh/i.test(v.lang));
      if(!voice)throw Error('未发现中文系统音色，请安装中文语音包或配置 TTS 服务');
      const utterance=new SpeechSynthesisUtterance(text);utterance.voice=voice;utterance.lang='zh-CN';utterance.rate=1;
      utterance.onend=()=>status('#audioStatus','朗读结束，可以回答。');
      utterance.onerror=e=>{if(!['canceled','interrupted'].includes(e.error))status('#audioStatus','朗读失败，请点击重播。',true)};
      speechSynthesis.speak(utterance);
    }
    status('#audioStatus','面试官正在说话……');
  }catch(e){status('#audioStatus',e.message,true)}
}
function currentSpeech(){const s=state.session;return s?.status==='ended'?s.report?.summary:s?.messages.filter(m=>m.role==='assistant').at(-1)?.text}
function autoSpeak(){if($('#autoSpeak').checked)speak(currentSpeech())}
$('#replayBtn').onclick=()=>speak(currentSpeech());$('#stopVoiceBtn').onclick=stopSpeech;

async function history(){
  const list=await api('/api/sessions');
  $('#history').innerHTML=list.length?list.map(s=>`<button class="secondary" data-sid="${s.id}">${esc(new Date(s.created*1000).toLocaleString())} · ${esc(s.role||labels[s.track]||'自定义岗位')} · ${s.status==='ended'?'已结束':s.status==='paused'?'已暂停':'进行中'}</button>`).join(''):'<span class="hint">还没有面试记录</span>';
  $$('[data-sid]').forEach(b=>b.onclick=()=>run(async()=>{state.session=await api('/api/sessions/'+b.dataset.sid);renderSession()}));
}
function invalidateResume(){state.resume=null;$('#evidenceCard').classList.add('hidden');status('#parseStatus','简历已更改，请重新解析。');updateControls()}
$('#file').onchange=()=>{const f=$('#file').files[0];if(f){$('#fileName').textContent=f.name;invalidateResume()}};
$('#resumeText').oninput=()=>{$('#file').value='';$('#fileName').textContent='将使用下方编辑的简历文字';invalidateResume()};
$$('[data-track]').forEach(b=>b.onclick=()=>{state.track=b.dataset.track;$('#role').value=labels[state.track];$$('[data-track]').forEach(x=>x.classList.toggle('active',x===b))});
$('#role').oninput=()=>{$$('[data-track]').forEach(x=>x.classList.remove('active'));state.track='custom'};
function chooseMode(mode){state.mode=mode;$('#demoMode').classList.toggle('active',mode==='demo');$('#liveMode').classList.toggle('active',mode==='live');updateControls()}
for(const mode of ['demo','live'])$('#'+mode+'Mode').onclick=()=>chooseMode(mode);
$('#parseBtn').onclick=()=>run(async()=>{
  state.resume=null;$('#evidenceCard').classList.add('hidden');
  status('#parseStatus','正在提取全文、识别章节和整理条目……');
  const file=$('#file').files[0], body={text:$('#resumeText').value};
  if(file){
    if(file.size>8*1024*1024)throw Error('文件最多8MB');
    const bytes=new Uint8Array(await file.arrayBuffer());let binary='';
    for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
    body.name=file.name;body.content=btoa(binary);
  }
  state.resume=await api('/api/resume',body);$('#resumeText').value=state.resume.text;
  $('#evidenceCard').classList.remove('hidden');renderEvidence();
  status('#parseStatus',`已整理 ${state.resume.cards.length} 个完整条目：${resumeKinds.slice(0,4).map(k=>k+' '+state.resume.summary[k]).join(' / ')}。请核对下方结果。`);
  $('#parseWarnings').textContent=state.resume.warnings.join('\n');
  $('#parseWarnings').classList.toggle('hidden',!state.resume.warnings.length);
  $('#evidenceCard').scrollIntoView({behavior:'smooth',block:'start'});
}, '#parseStatus');
function renderEvidence(){
  const cards=state.resume.cards;
  const kinds=resumeKinds.filter(k=>resumeKinds.indexOf(k)<4||cards.some(c=>c.kind===k));
  $('#evidence').innerHTML=kinds.map(kind=>{
    const entries=cards.filter(c=>c.kind===kind);
    return `<section class="resume-group"><div class="group-heading"><h3>${kind==='其他'?'其他 / 待归类':kind} <span>${entries.length}</span></h3><button class="secondary" data-add-kind="${kind}">补充${kind}</button></div>${entries.length?entries.map((c,i)=>`<div class="evidence"><div class="meta"><b>${kind} ${i+1}</b><select aria-label="条目分类" data-kind="${c.id}">${resumeKinds.map(k=>`<option ${k===c.kind?'selected':''}>${k}</option>`).join('')}</select><button class="secondary" data-remove="${c.id}">移除</button></div><small>${c.method==='heading'?'按章节标题识别':c.method==='inferred'?'按内容线索推断，请核对':c.method==='manual'?'手动添加':'未能确定分类，请核对'}${c.source?(state.resume.source_format==='pdf'?' · PDF第'+c.source.pages.join('、')+'页':' · 提取文本第'+c.source.lines[0]+'行'):''}</small><textarea aria-label="${kind}内容" data-text="${c.id}" maxlength="12000">${esc(c.text)}</textarea></div>`).join(''):`<p class="hint">未识别到${kind}。如果简历中有这部分，请补充或从其他分类移入。</p>`}</section>`;
  }).join('');
  $$('[data-remove]').forEach(b=>b.onclick=()=>{syncCards();state.resume.cards=state.resume.cards.filter(c=>c.id!==b.dataset.remove);renderEvidence();resetConfirmation()});
  $$('[data-add-kind]').forEach(b=>b.onclick=()=>{syncCards();if(cards.length>=100)return;state.resume.cards.push({id:crypto.randomUUID(),text:'',kind:b.dataset.addKind,confirmed:false,method:'manual'});renderEvidence();resetConfirmation()});
  $$('[data-kind]').forEach(select=>select.onchange=()=>{syncCards();renderEvidence();resetConfirmation()});
  $$('[data-text]').forEach(t=>t.oninput=resetConfirmation);
  resetConfirmation();
}
function resetConfirmation(){$('#confirmResume').checked=false}
function syncCards(){state.resume.cards=state.resume.cards.map(c=>({...c,text:$(`[data-text="${c.id}"]`).value,kind:$(`[data-kind="${c.id}"]`).value,confirmed:$('#confirmResume').checked}))}
$('#startBtn').onclick=()=>run(async()=>{
  syncCards();
  if(!state.resume.cards.length||state.resume.cards.some(c=>!c.confirmed||!c.kind||!c.text.trim()))throw Error('请填写或移除空条目，并勾选“已核对分类和内容”。');
  status('#startStatus','正在准备面试……');stopSpeech();
  state.session=await api('/api/sessions',{cards:state.resume.cards,track:state.track,role:$('#role').value,jd:$('#jd').value,style:$('#style').value,minutes:+$('#minutes').value,mode:state.mode});
  $('#answer').value='';renderSession();autoSpeak();
}, '#startStatus');
$('#startFromResume').onclick=()=>$('#startBtn').click();
function renderSession(){
  const s=state.session;
  $('#setup').classList.add('hidden');$('#interview').classList.toggle('hidden',s.status==='ended');$('#review').classList.toggle('hidden',s.status!=='ended');
  $('#sessionTitle').textContent=s.role||labels[s.track]||'自定义岗位';$('#sessionMeta').textContent=`${s.style} · ${s.mode==='demo'?'规则演示（非 AI）':'联网 AI'}`;
  $('#pauseBtn').textContent=s.status==='paused'?'继续':'暂停';
  $('#anchor').textContent=s.plan[s.main_index]?.anchor||'当前问题没有绑定特定经历。';
  $('#jobFocus').textContent=s.profile?.requirements?.length?s.profile.requirements.map(r=>`${r.name}：${r.quote}`).join('\n'):s.profile?.focus||'暂无岗位分析';
  $('#stage').textContent=`${s.plan[s.main_index]?.topic} · 追问 ${s.depth} / ${s.plan[s.main_index]?.follow_limit}`;
  $('#sessionError').innerHTML=s.error?`<div class="error">${esc(s.error)}</div>`:'';
  $('#chat').innerHTML=s.messages.map(m=>`<div class="msg ${m.role==='user'?'user':''}"><b>${m.role==='user'?'你':'面试官'} · ${esc(m.topic)}</b><div>${esc(m.text)}</div></div>`).join('');
  $('#chat').scrollTop=$('#chat').scrollHeight;
  $('#exportMd').href='/api/sessions/'+s.id+'/export?format=md';$('#exportJson').href='/api/sessions/'+s.id+'/export?format=json';
  if(s.status==='ended')renderReport();updateClock();if(!state.busy)updateControls();
}
function updateClock(){const s=state.session;if(!s)return;const elapsed=s.elapsed+(s.status==='active'?Math.max(0,Date.now()/1000-s.last_tick):0);$('#timer').textContent=`${fmt(elapsed)} / ${fmt(s.minutes*60)}`;$('#meter').style.width=Math.min(100,elapsed/(s.minutes*60)*100)+'%'}
setInterval(updateClock,1000);
async function mutate(action,text=''){
  try{state.session=await api('/api/sessions/'+state.session.id+'/'+action,{version:state.session.version,text});renderSession()}
  catch(e){state.session=await api('/api/sessions/'+state.session.id);renderSession();throw e}
}
$('#sendBtn').onclick=()=>run(async()=>{
  const answer=$('#answer').value.trim();if(!answer)throw Error('请先输入或录制回答');
  stopSpeech();await mutate('answer',answer);$('#answer').value='';autoSpeak();
  if(state.session.status==='ended')await makeReport();
});
$('#retryBtn').onclick=()=>run(async()=>{await mutate('retry');autoSpeak()});
$('#pauseBtn').onclick=()=>run(async()=>{stopSpeech();await mutate(state.session.status==='paused'?'resume':'pause')});
$('#endBtn').onclick=()=>run(async()=>{if($('#answer').value.trim()&&!confirm('当前未提交的文字不会进入报告，仍要结束吗？'))return;stopSpeech();await mutate('end');await makeReport()});
async function makeReport(){status('#audioStatus','正在整理逐题反馈……');await mutate('report');renderReport();status('#audioStatus','复盘已生成，可导出记录或重播总结。');autoSpeak()}
$('#reportBtn').onclick=()=>run(makeReport);
function renderReport(){
  const r=state.session.report;$('#reviewSummary').textContent=r?.summary||'面试记录已保存，可生成复盘或直接导出。';
  $('#report').innerHTML=r?r.items.map(x=>`<div class="reportitem"><div class="score">${x.score?x.score+'/5':'未评分'} <small>训练等级</small></div><blockquote>“${esc(x.quote)}”</blockquote><div class="kv"><b>做得好</b><span>${esc(x.strength)}</span><b>证据缺口</b><span>${esc(x.gap)}</span><b>下一次</b><span>${esc(x.action)}</span></div></div>`).join('')+`<div class="notice"><b>下次练习</b><br>${r.actions.map(esc).join('<br>')}</div>`:'<div class="notice">尚未生成复盘。模型失败后可重试，原始转录不受影响。</div>';
}
$('#backBtn').onclick=()=>run(async()=>{stopSpeech();state.session=null;$('#answer').value='';$('#review').classList.add('hidden');$('#interview').classList.add('hidden');$('#setup').classList.remove('hidden');await history()});
$('#deleteBtn').onclick=()=>run(async()=>{if(!confirm('删除这场面试的本地记录和报告？'))return;stopSpeech();await api('/api/sessions/'+state.session.id,null,'DELETE');state.session=null;$('#review').classList.add('hidden');$('#interview').classList.add('hidden');$('#setup').classList.remove('hidden');await history()});

async function transcribeBlob(){
  status('#voiceStatus','正在识别录音……');
  const res=await fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':state.blob.type},body:state.blob});const d=await res.json();
  if(!res.ok)throw Error(d.detail||'识别失败');
  $('#answer').value=($('#answer').value+' '+d.text).trim();status('#voiceStatus','请核对技术名词后提交。');
}
$('#retryAudioBtn').onclick=()=>run(transcribeBlob);
$('#recordBtn').onclick=async()=>{
  if(state.busy)return;
  stopSpeech();
  if(state.recorder){state.recorder.stop();return}
  if(state.recognition){state.recognition.stop();return}
  if(!state.config.stt){
    const Recognizer=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!Recognizer){status('#voiceStatus','浏览器不支持语音识别，请配置 STT API；仍可文字回答。',true);return}
    const recog=new Recognizer();state.recognition=recog;recog.lang='zh-CN';recog.continuous=true;recog.interimResults=false;
    recog.onresult=e=>{for(let i=e.resultIndex;i<e.results.length;i++)if(e.results[i].isFinal)$('#answer').value+=e.results[i][0].transcript};
    recog.onerror=e=>status('#voiceStatus','浏览器识别失败：'+e.error+'。可配置 STT API 或文字回答。',true);
    recog.onend=()=>{state.recognition=null;$('#recordBtn').textContent='● 录音';updateControls()};
    try{recog.start();$('#recordBtn').textContent='■ 停止识别';status('#voiceStatus','浏览器识别中，可能通过浏览器服务联网处理。');updateControls()}
    catch(e){state.recognition=null;status('#voiceStatus',e.message,true)}
    return;
  }
  try{
    state.stream=await navigator.mediaDevices.getUserMedia({audio:true});state.chunks=[];
    const mime=['audio/webm;codecs=opus','audio/ogg;codecs=opus'].find(t=>MediaRecorder.isTypeSupported(t));
    if(state.config.provider==='bailian'&&!mime)throw Error('百炼录音请使用支持 WebM/OGG 的 Chrome 或 Edge 浏览器');
    const recorder=new MediaRecorder(state.stream,mime?{mimeType:mime}:{});state.recorder=recorder;
    const limit=setTimeout(()=>{if(recorder.state==='recording')recorder.stop()},180000);
    recorder.ondataavailable=e=>state.chunks.push(e.data);
    recorder.onstop=async()=>{
      clearTimeout(limit);state.stream.getTracks().forEach(t=>t.stop());state.stream=null;state.recorder=null;
      state.blob=new Blob(state.chunks,{type:recorder.mimeType||'audio/webm'});
      if(state.blobUrl)URL.revokeObjectURL(state.blobUrl);state.blobUrl=URL.createObjectURL(state.blob);
      $('#downloadAudio').href=state.blobUrl;$('#downloadAudio').download='interview-answer.'+(state.blob.type.includes('mp4')?'m4a':state.blob.type.includes('ogg')?'ogg':'webm');$('#downloadAudio').classList.remove('hidden');
      $('#recordBtn').textContent='● 录音';await run(transcribeBlob);
    };
    recorder.start();$('#recordBtn').textContent='■ 停止录音';status('#voiceStatus','录音中（单次最多3分钟）');updateControls();
  }catch(e){state.stream?.getTracks().forEach(t=>t.stop());state.recorder=null;status('#voiceStatus','无法录音：'+e.message,true);updateControls()}
};
window.addEventListener('beforeunload',()=>{stopSpeech();state.stream?.getTracks().forEach(t=>t.stop());state.recognition?.abort()});
(async()=>{try{state.config=await api('/api/config');$('#modePill').textContent=state.config.llm?'本地数据 · AI 已配置':'本地数据 · 规则演示可用';renderReadiness();await history();updateControls();window.speechSynthesis?.getVoices()}catch(e){status('#audioStatus',e.message,true)}})();

function renderReadiness(){
  const c=state.config;
  $('#readiness').innerHTML=c.llm?
    '<b>AI 配置已填写</b> · 首次面试将验证连接。语音识别：'+(c.stt?'云端已配置':'浏览器备用')+'；朗读：'+(c.tts?'云端已配置':'系统音色'):
    '<b>当前只能运行规则演示，尚未接入真实 AI。</b><br>简历解析可正常使用。要进行 AI 追问，请在项目 .env 中填写 '+esc((c.missing?.llm||['LLM_URL','LLM_KEY','LLM_MODEL']).join('、'))+'，然后重启服务并刷新页面。';
  $('#liveMode').title=c.llm?'根据简历与JD动态追问':'尚未配置LLM，请先填写 .env 并重启';
  if(c.llm)chooseMode('live');
}
