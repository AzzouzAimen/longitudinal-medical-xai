'use strict';
const {esc, finite, format, time, dateTime, signalPlot, RequestGate, Drafts} = ViewerCore;
const $ = id => document.getElementById(id);
let storage;
try { storage = window.sessionStorage; } catch (_) {}
const drafts = new Drafts(storage), eventGate = new RequestGate(), patientGate = new RequestGate(), patientListGate = new RequestGate();
const feedbackCache = new Map(), feedbackMessages = new Map(), feedbackVersions = new Map(), saves = new Set();
let D = null, activeTab = 'overview';
const sourceNames = {current:'Current', history:'History · PDM', both:'Current + history', unknown:'Source not classified'};
const draftKey = data => `${data.event.prediction_id}:${data.narrative?.narrative_id ?? 'none'}`;
const noData = message => `<div class="empty">${esc(message || 'No data available.')}</div>`;
const flagText = flag => flag === true ? 'Flagged' : flag === false ? 'Not flagged' : 'No data';
const decision = event => event.is_anomaly === true ? 'Anomaly' : event.is_anomaly === false ? 'Normal' : 'No data';
async function getJSON(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch (_) { throw Error('The service returned an unreadable response.'); }
  if (!response.ok) throw Error(data.error || 'Request failed.');
  return data;
}
function status(message, kind = '') { $('status').textContent = message; $('status').className = `connection ${kind}`; }
function clearView(message, error = false) {
  D = null; $('tabs').innerHTML = ''; $('eventHeading').innerHTML = '<p>Choose an event to inspect its evidence.</p>';
  $('view').innerHTML = `<div class="empty page-empty ${error?'inline-error':''}">${esc(message)}</div>`;
  $('view').removeAttribute('aria-labelledby');
}
async function loadPatients() {
  const token=patientListGate.begin();
  status('Connecting…'); $('patient').disabled = true;
  try {
    const patients = await getJSON('/api/patients');
    if(!patientListGate.current(token))return;
    $('patient').innerHTML = patients.map(p => `<option value="${esc(p.patient_code)}">${esc(p.patient_code)} · ${esc(p.modality || 'unknown')}</option>`).join('');
    if (!patients.length) { clearView('No patients are stored in the PDM.'); status('No patient data'); return; }
    $('patient').disabled = false;
    $('patient').value = patients.find(p => p.patient_code === 'SLEEP_SC401')?.patient_code || patients[0].patient_code;
    await loadEvents();
  } catch (error) { if(patientListGate.current(token)) { clearView(error.message,true); status('Connection unavailable','error'); } }
}
async function loadEvents() {
  const token = patientGate.begin(), patient = $('patient').value;
  eventGate.begin(); $('event').disabled = true; $('event').innerHTML = '<option>Loading events…</option>';
  clearView('Loading patient events…'); status('Loading…'); activeTab = 'overview';
  try {
    const events = await getJSON(`/api/patient/${encodeURIComponent(patient)}/events`);
    if (!patientGate.current(token)) return;
    $('event').innerHTML = events.map(e => `<option value="${e.prediction_id}">${esc(e.recording_id)} · ${time(e.start_time)} UTC · Window ${e.window_index} · ${decision(e)}${e.has_narrative?' · narrative':''}</option>`).join('');
    if (!events.length) { clearView('No analyzed events for this patient.'); status('No event data'); return; }
    const selected = events.find(e => e.prediction_id === 7197) || events.find(e => e.has_narrative && e.is_anomaly) || events.find(e => e.is_anomaly) || events[0];
    $('event').value = selected.prediction_id; $('event').disabled = false;
    await loadEvent(selected.prediction_id);
  } catch (error) { if (patientGate.current(token)) { clearView(error.message,true); status('Connection unavailable','error'); } }
}
async function loadEvent(id) {
  const token = eventGate.begin(); clearView('Loading event evidence…'); status('Loading…');
  try {
    const data = await getJSON(`/api/event/${id}/evidence`);
    if (!eventGate.current(token)) return;
    D = data;
    if (!tabDefinitions().some(([key]) => key === activeTab)) activeTab = 'overview';
    render(); status('Connected to PDM','connected');
    loadFeedback(data);
  } catch (error) { if (eventGate.current(token)) { clearView(error.message,true); status('Event unavailable','error'); } }
}
function tabDefinitions() {
  return [['overview','Overview'], ['summary','Summary'], ['feedback','Feedback'], ...D.channels.map((channel,index)=>[`channel-${index}`,channel.name])];
}
function icon(name, extra = '') {
  return `<svg class="icon ${extra}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true" focusable="false"><use href="/static/icons/heroicons.svg#${name}"></use></svg>`;
}
function renderTabs() {
  const dirty = drafts.get(draftKey(D)).notes.trim().length > 0;
  const icons = {overview:'squares-2x2', summary:'document-text', feedback:'pencil-square'};
  const button = ([key,label]) => `<button type="button" role="tab" id="tab-${key}" data-tab="${key}" class="${key==='feedback'&&dirty?'has-draft':''}" ${key==='feedback'?`aria-label="${dirty?'Feedback (unsaved draft)':'Feedback'}"`:''} aria-controls="view" aria-selected="${activeTab===key}" tabindex="${activeTab===key?0:-1}">${icons[key]?icon(icons[key]):''}<span>${esc(label)}</span></button>`;
  const definitions = tabDefinitions();
  $('tabs').innerHTML = `<div class="primary-tabs">${definitions.slice(0,3).map(button).join('')}</div>${D.channels.length?`<div class="channel-navigation"><span class="channel-label">${icon('chart-bar-square')}Channels</span><div class="channel-tabs">${definitions.slice(3).map(button).join('')}</div></div>`:''}`;
  $('view').setAttribute('aria-labelledby',`tab-${activeTab}`);
}
function switchTab(key, focus = false) {
  if (!D || !tabDefinitions().some(([value])=>value===key)) return;
  activeTab=key; render();
  if (focus) $(`tab-${key}`).focus();
}
function render() {
  if (!D) return;
  const event = D.event;
  $('eventHeading').innerHTML = `<div><h2>${esc(event.patient_code)} <span class="muted">/</span> ${esc(event.source_recording_name || event.recording_id)}</h2><p>Prediction #${event.prediction_id} · ${format(event.duration_seconds,0)}-second window · ${dateTime(event.start_time)} · ${esc(event.modality)}</p></div><span class="result ${event.is_anomaly===true?'anomaly':event.is_anomaly===false?'normal':''}">${decision(event)}</span>`;
  renderTabs();
  $('view').innerHTML = activeTab==='overview' ? overview(D) : activeTab==='summary' ? summary(D) : activeTab==='feedback' ? feedbackView(D) : channelView(D,D.channels[Number(activeTab.slice(8))]);
  if (activeTab==='feedback') bindFeedback(D);
}
const sectionIcons = {'Current evidence':'chart-bar-square','Historical context':'clock','Evidence-grounded narrative':'document-text','Full narrative':'document-text','Event':'information-circle','Channel evidence':'chart-bar-square','Auxiliary channels':'circle-stack','History & provenance':'link','Channel measurements':'chart-bar-square','Channel-specific narrative':'document-text','Shared context':'clock','Feedback':'pencil-square','Saved feedback':'document-text'};
const sectionTones = {'Current evidence':'tone-current','Channel evidence':'tone-current','Channel measurements':'tone-current','Event':'tone-current','Historical context':'tone-history','History & provenance':'tone-history','Shared context':'tone-history','Evidence-grounded narrative':'tone-narrative','Full narrative':'tone-narrative','Channel-specific narrative':'tone-narrative'};
function section(title, subtitle, content, extra = '') {
  return `<section class="section ${sectionTones[title] || ''} ${extra}"><div class="section-head">${icon(sectionIcons[title] || 'chart-bar-square')}<div><h2>${esc(title)}</h2>${subtitle?`<p class="sub ${subtitle==='Flagged'?'flag-state':''}">${esc(subtitle)}</p>`:''}</div></div>${content}</section>`;
}
function disclosure(title, subtitle, content, open = false) {
  return `<details class="section disclosure-section ${sectionTones[title] || ''}" ${open?'open':''}><summary>${icon(sectionIcons[title] || 'information-circle')}<span><h2>${esc(title)}</h2>${subtitle?`<span class="sub">${esc(subtitle)}</span>`:''}</span>${icon('chevron-down','disclosure-chevron')}</summary><div class="disclosure-body">${content}</div></details>`;
}
function viewIntro(title, description) {
  return `<div class="view-intro"><h2>${esc(title)}</h2><p>${esc(description)}</p></div>`;
}
function evidencePlot(points, event, options = {}) {
  const width = Math.min(720, Math.max(320, (window.innerWidth || 1440) - 80));
  return signalPlot(points, event, {...options, width});
}
function fact(label, value, danger = false) { return `<div><span class="fact-label">${esc(label)}</span><span class="fact-value ${danger?'danger':''}">${esc(value)}</span></div>`; }
function table(rows) { return rows.length ? `<div class="table-scroll"><table class="details-table"><tbody>${rows.map(([label,value])=>`<tr><th scope="row">${esc(label)}</th><td>${esc(value ?? 'No data')}</td></tr>`).join('')}</tbody></table></div>` : noData('No measurements stored.'); }
function claimsView(claims, emptyMessage='No claims stored for this view.') {
  if (!claims.length) return noData(emptyMessage);
  return `<ol class="claim-list">${claims.map(claim=>`<li class="claim"><div class="claim-meta"><span class="source-label ${esc(claim.source)}">${esc(sourceNames[claim.source] || sourceNames.unknown)}</span><span>${esc(claim.claim_id)}</span></div><p>${esc(claim.text)}</p>${claim.evidence_ids.length?`<details class="citations"><summary>Evidence · ${claim.evidence_ids.length} reference${claim.evidence_ids.length===1?'':'s'}</summary>${claim.evidence_ids.map(id=>`<code>${esc(id)}</code>`).join('')}</details>`:noData('No evidence references stored.')}</li>`).join('')}</ol>`;
}
function narrativeView(narrative, claims) {
  if (!narrative) return noData('No narrative has been stored for this event.');
  if (narrative.claims.length) return claimsView(claims || narrative.claims);
  return `<div class="legacy-narrative">${esc(narrative.text)}</div><p class="small-note">This stored narrative has no structured claim-level citations.</p>`;
}
function attentionView(data) {
  const attention = data.attention;
  if (!attention.points.length) return noData('No temporal attention data stored for this window.');
  return `${evidencePlot(attention.points,data.event,{label:'Shared temporal attention',unit:'Relevance weight',compact:true,attention:true,markers:[{time:attention.peak_time,label:'Shared attention peak'}]})}<p class="plot-caption">${attentionCaption(data)}<span>Window-level relevance</span></p>`;
}
function attentionCaption(data) {
  return data.attention.peak_time ? `<span class="attention-key">Shared attention peak · ${time(data.attention.peak_time)} UTC</span>` : '<span>No shared attention peak stored</span>';
}
function channelPlot(data, channel) {
  return evidencePlot(channel?.signal || [], data.event, {label:channel?.name || 'Signal', unit:channel?.unit, markers:[{time:data.attention.peak_time,label:'Shared attention peak'}], regions:channel?.regions || []});
}
function historyView(data) {
  const previous = data.history.previous_recording;
  let body, recording = '';
  if (previous) {
    body = `<div class="history-stat"><strong class="${previous.anomalies>0?'focus-value':''}">${format(previous.rate)}%</strong><span>of retained windows flagged on the previous ${data.event.modality==='sleep'?'night':'recording'}</span></div><div class="bar ${previous.anomalies>0?'flagged-history':''}"><i style="width:${Math.max(0,Math.min(100,Number(previous.rate)||0))}%"></i></div><p class="history-count"><strong class="${previous.anomalies>0?'focus-value':''}">${format(previous.anomalies,0)} flagged</strong> / ${format(previous.windows,0)} retained windows</p>`;
    recording = `<div class="timeline"><div>Previous ${data.event.modality==='sleep'?'night':'recording'}<small>${esc(previous.recording_id)}</small></div>${icon('arrow-right')}<div>Current event<small>${esc(data.event.recording_id)}</small></div></div>`;
  } else body = noData(`No previous ${data.event.modality==='sleep'?'night':'recording'} data available for this patient.`);
  const recent = data.history.recent_events;
  const recentTable = recent.length ? `<h3>Recent earlier events</h3><div class="table-scroll"><table><thead><tr><th>Time (UTC)</th><th>Prediction</th><th>Detector</th></tr></thead><tbody>${recent.map(event=>`<tr><td>${time(event.start_time)}</td><td>#${event.prediction_id}</td><td>${decision(event)}</td></tr>`).join('')}</tbody></table></div>` : noData('No earlier events stored.');
  return `${body}<details class="plot-details"><summary>${icon('clock')}Recording details and recent events ${icon('chevron-down','disclosure-chevron')}</summary><div class="detail-content">${recording}${recentTable}<p class="small-note">Same-patient historical context. Counts describe retained windows, not the full recording.</p></div></details>`;
}
function overview(data) {
  const event=data.event, ranked=[...data.channels].sort((a,b)=>(b.contribution ?? -1)-(a.contribution ?? -1));
  const top=ranked[0], flags=data.channels.filter(channel=>channel.flagged===true).length;
  const main = top && finite(top.contribution) ? `${top.name} · ${format(top.contribution)}%` : 'No channel contribution data';
  const knownFlags=data.channels.filter(channel=>typeof channel.flagged==='boolean').length;
  const flagsLabel=!knownFlags?'No data':knownFlags===data.channels.length?`${flags} / ${knownFlags} modeled channels`:`${flags} / ${knownFlags} channels with flag data · ${data.channels.length-knownFlags} unavailable`;
  const context=`<div class="facts">${fact(event.modality==='sleep'?'Recorded sleep stage':'Recorded annotation',event.official_label || 'No data')}${fact('Attention peak',data.attention.peak_time?`${time(data.attention.peak_time)} UTC`:'No data',!!data.attention.peak_time)}</div><p class="small-note">The recorded annotation is context, not detector ground truth or a diagnosis.</p>${attentionView(data)}`;
  const current = `<div class="detector-reading"><div><span class="fact-label">Reconstruction error</span><strong class="${event.is_anomaly===true && finite(event.reconstruction_error)?'focus-value':''}">${format(event.reconstruction_error)}</strong></div><span class="comparison-divider">/</span><div><span class="fact-label">Detector threshold</span><strong>${format(event.anomaly_threshold)}</strong></div></div>${channelPlot(data,top)}<p class="plot-caption"><span>${esc(top?.name || 'Signal unavailable')}</span>${attentionCaption(data)}</p><div class="facts">${fact('Main contributor',main,top?.flagged===true && finite(top.contribution))}${fact('Channel flags',flagsLabel,flags>0)}</div><details class="plot-details"><summary>${icon('information-circle')}Recorded context & shared attention ${icon('chevron-down','disclosure-chevron')}</summary><div class="detail-content">${context}</div></details>`;
  const claims=data.narrative?.claims || [];
  const selected=[];
  const detector=claims.find(c=>c.source==='current' && c.evidence_ids.some(id=>id.endsWith(':detector'))) || claims.find(c=>c.source==='current');
  if(detector)selected.push(detector);
  for (const type of ['history','both']) { const claim=claims.find(c=>c.source===type); if(claim)selected.push(claim); }
  if(!selected.length)selected.push(...claims.slice(0,3));
  const narrative = `${narrativeView(data.narrative,selected)}<div class="section-foot"><span class="muted">${claims.length ? `${claims.length} stored claims` : 'Stored narrative'}</span><button class="text-link" data-go="summary">Full narrative ${icon('arrow-right')}</button></div>`;
  const fusion=`<div class="fusion-connector"><div class="fusion-direction" role="img" aria-label="Current evidence and patient history combine through evidence fusion into the narrative"><span>EVIDENCE FUSION</span><div class="fusion-line" aria-hidden="true">${icon('arrow-right')}</div></div><p>The LLM relates supplied evidence only; interpretation remains with the human reviewer.</p></div>`;
  return `${viewIntro('Evidence overview','Review the detector result, then compare it with the patient’s earlier evidence.')}<div class="overview-grid overview-diagram"><div class="stack evidence-inputs">${section('Current evidence',`${top?.name || 'Signal unavailable'} · ${format(event.duration_seconds,0)}-second window`,current,'current-section')}${section('Historical context','Earlier evidence from the same patient',historyView(data))}</div>${fusion}<div class="stack evidence-output">${section('Evidence-grounded narrative','Key claims from the stored narrative',narrative,'interpretation-section')}</div></div>`;
}
function channelClaims(data, channel) {
  return (data.narrative?.claims || []).filter(claim=>claim.evidence_ids.some(id=>id.includes(channel.name) || (channel.signal_id && id.endsWith(`signal:${channel.signal_id}`))) || claim.text.includes(channel.name));
}
function channelView(data,channel) {
  if (!channel) return noData('Channel unavailable.');
  const markers=(channel.regions || []).filter(region=>region.temporal_start && region.temporal_end && new Date(region.temporal_end)>new Date(region.temporal_start));
  const contribution = `<div class="contribution ${channel.flagged===true && finite(channel.contribution)?'flagged-contribution':''}"><span>Residual contribution</span><div class="bar"><i style="width:${finite(channel.contribution)?Math.max(0,Math.min(100,channel.contribution)):0}%"></i></div><strong>${finite(channel.contribution)?`${format(channel.contribution)}%`:'No data'}</strong></div>`;
  const waveform = `${channelPlot(data,channel)}<p class="plot-caption"><span>Recorded signal · ${esc(channel.unit || 'unit unavailable')}</span><span>Dashed line: shared attention peak</span></p>${contribution}<details class="plot-details"><summary>${icon('chart-bar-square')}Reconstruction and residual ${icon('chevron-down','disclosure-chevron')}</summary><div class="detail-content"><h3>Reconstruction</h3>${evidencePlot(channel.reconstruction,data.event,{label:`${channel.name} reconstruction`,unit:channel.reconstruction_unit})}<h3>Residual</h3>${evidencePlot(channel.residual,data.event,{label:`${channel.name} residual`,unit:channel.residual_unit})}</div></details><details class="plot-details"><summary>${icon('clock')}Shared temporal attention ${icon('chevron-down','disclosure-chevron')}</summary><div class="detail-content">${attentionView(data)}<p class="small-note">Attention has no channel owner and is not a probability.</p></div></details><details class="plot-details"><summary>${icon('information-circle')}How to read this evidence ${icon('chevron-down','disclosure-chevron')}</summary><div class="detail-content"><p class="small-note">${markers.length?'Shading shows a stored residual region.':'No anomalous interval is stored for this channel.'} Channel residuals come from one shared model; they are not independent detectors.</p></div></details>`;
  const specific=channelClaims(data,channel), shared=(data.narrative?.claims || []).filter(c=>!specific.includes(c) && (c.section==='current_event' || c.source==='both'));
  return `${viewIntro(channel.name,'Inspect the recorded signal and its contribution to the shared model result.')}<div class="two-column signal-layout"><div class="stack">${section('Current evidence',flagText(channel.flagged),waveform)}${disclosure('Channel measurements','Stored XAI features',table(Object.entries(channel.features).map(([key,value])=>[key.replaceAll('_',' '),format(value,5)])))}</div><div class="stack">${section('Channel-specific narrative','Stored claims referencing this channel',claimsView(specific,'No channel-specific claims stored.'))}${disclosure('Shared context','Current event and historical synthesis',claimsView(shared,'No shared context claims stored.'))}</div></div>`;
}
function groupedNarrative(narrative) {
  if(!narrative || !narrative.claims.length)return narrativeView(narrative);
  const titles={evidence_grounded_summary:'Overall interpretation',current_event:'Current event',model_and_xai_evidence:'Model & channel evidence',previous_monitoring_history:'Patient history',recorded_clinical_context:'Recorded context'};
  const groups=new Map();
  for(const claim of narrative.claims) {
    const key=claim.section || 'other';
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(claim);
  }
  const order=[...Object.keys(titles),...Array.from(groups.keys()).filter(key=>!(key in titles))].filter(key=>groups.has(key));
  return `<div class="grouped-narrative">${order.map((key,index)=>`<details class="narrative-group" ${index===0?'open':''}><summary><span>${esc(titles[key] || key.replaceAll('_',' '))}</span><span class="group-count">${groups.get(key).length} claim${groups.get(key).length===1?'':'s'}</span>${icon('chevron-down','disclosure-chevron')}</summary><div class="group-content">${claimsView(groups.get(key))}</div></details>`).join('')}</div>`;
}
function summary(data) {
  const event=data.event, claims=data.narrative?.claims || [], n=data.narrative;
  const details=table([['Patient',event.patient_code],['Recording',event.recording_id],['Prediction',`#${event.prediction_id}`],['Decision',decision(event)],['Error / threshold',`${format(event.reconstruction_error,4)} / ${format(event.anomaly_threshold,4)}`],['Attention peak',data.attention.peak_time?dateTime(data.attention.peak_time):'No data'],['Recorded annotation',event.official_label],['Annotation source',event.annotation_source],['Model',`${event.model_name || 'No data'} · ${event.model_version || 'No data'}`]]);
  const channels=data.channels.length?`<div class="table-scroll"><table><thead><tr><th>Channel</th><th>Share</th><th>Flag</th></tr></thead><tbody>${data.channels.map(c=>`<tr><td>${esc(c.name)}</td><td>${finite(c.contribution)?`${format(c.contribution)}%`:'No data'}</td><td class="${c.flagged===true?'flag':''}">${flagText(c.flagged)}</td></tr>`).join('')}</tbody></table></div>`:noData('No modeled channel data.');
  const auxiliary=data.auxiliary.length?`<ul class="auxiliary-list">${data.auxiliary.map(c=>`<li>${esc(c.name)} · ${format(c.sampling_rate_hz,0)} Hz · ${c.signal.length} stored points</li>`).join('')}</ul>`:noData('No auxiliary channel data.');
  const history=data.history.previous_recording;
  const provenance=table([['Previous '+(event.modality==='sleep'?'night':'recording'),history?.recording_id],['Windows / anomalies',history?`${format(history.windows,0)} / ${format(history.anomalies,0)} (${format(history.rate)}%)`:'No data'],['Narrative',n?`#${n.narrative_id}`:'No data'],['Generated',n?dateTime(n.generated_at):'No data'],['LLM model',n?.llm_model_version],['Claims',claims.length?`${claims.length} (${claims.filter(c=>c.source==='current').length} current, ${claims.filter(c=>c.source==='history').length} history, ${claims.filter(c=>c.source==='both').length} both)`:'No structured claims'],['Package hash',data.provenance.package_hash || 'No data']]);
  const refs=data.provenance.narrative_evidence;
  return `${viewIntro('Summary','Read the stored interpretation, then expand a section to inspect its supporting details.')}<div class="two-column summary-layout">${section('Full narrative',n?`PDM narrative #${n.narrative_id}`:'No narrative stored',`${groupedNarrative(n)}<p class="small-note">Detection is performed by the analytical model; the LLM organises evidence. This demonstration does not establish clinical utility.</p><div class="section-foot"><button class="text-link" data-go="feedback">${icon('pencil-square')}Add feedback on this narrative</button></div>`)}<div class="stack">${disclosure('Event','Stored detector result and recorded context',details)}${section('Channel evidence','Shared-model residual contributions',channels)}${disclosure('Auxiliary channels','Recorded context; not modeled reconstruction inputs',auxiliary)}${disclosure('History & provenance','Evidence traceability',`${provenance}<p class="small-note">${esc(data.provenance.package_hash_note)}</p><details class="plot-details"><summary>Narrative evidence references · ${refs.length}</summary>${refs.length?`<pre class="raw-package">${esc(JSON.stringify(refs,null,2))}</pre>`:noData('No evidence reference rows stored.')}</details><details class="plot-details"><summary>Viewer data</summary><p class="small-note">Current database view; not a frozen generation package.</p><pre class="raw-package">${esc(JSON.stringify(data,null,2))}</pre></details>`)}</div></div>`;
}
function savedFeedbackView(cached) {
  if(!cached)return noData('Loading saved feedback…');
  const notes=cached?.notes || [];
  const warning=cached.error?`<p class="inline-error">${esc(cached.error)}</p><button class="text-link" data-retry-feedback>Retry loading feedback</button>`:'';
  return warning+(notes.length?notes.map(note=>`<article class="feedback-note"><strong>${esc(note.reviewer || 'Unnamed reviewer')} · ${esc(note.subject)}${note.channel?` / ${esc(note.channel)}`:''}</strong><p>${esc(note.notes)}</p><div class="meta">${dateTime(note.created_at)} · Event #${note.prediction_id}${note.narrative_id?` · Narrative #${note.narrative_id}`:' · Event-level note'}</div></article>`).join(''):noData(cached.error?'Saved notes could not be loaded.':'No feedback saved for this event yet.'));
}
function feedbackView(data) {
  const key=draftKey(data), draft=drafts.get(key), cached=feedbackCache.get(data.event.prediction_id), message=feedbackMessages.get(key);
  const saved=savedFeedbackView(cached);
  return `<div class="feedback-grid">${section('Feedback','Your review of the evidence and narrative',`<p class="small-note">Event #${data.event.prediction_id}${data.narrative?` · Reviewing narrative #${data.narrative.narrative_id}`:' · No narrative stored'}</p><form id="feedbackForm"><div class="form-row"><div><label for="feedbackSubject">Feedback on</label><select id="feedbackSubject"><option value="overall">Overall result</option><option value="narrative" ${!data.narrative?'disabled':''}>Narrative</option><option value="channel" ${!data.channels.length?'disabled':''}>Channel evidence</option></select></div><div><label for="feedbackReviewer">Reviewer name <span class="muted">(optional)</span></label><input id="feedbackReviewer" maxlength="128" autocomplete="name" value="${esc(draft.reviewer)}"></div></div><div id="feedbackChannelField" ${draft.subject==='channel'?'':'hidden'}><label for="feedbackChannel">Channel</label><select id="feedbackChannel">${data.channels.map(c=>`<option value="${esc(c.name)}">${esc(c.name)}</option>`).join('')}</select></div><label for="feedbackNotes">Notes</label><textarea id="feedbackNotes" maxlength="10000" required placeholder="What needs clarification, correction, or further review?">${esc(draft.notes)}</textarea><p class="field-hint">Plain text · <span id="feedbackCount">${draft.notes.length.toLocaleString()}</span> / 10,000 characters. Drafts stay in this browser tab until saved.</p><div class="feedback-actions"><span class="muted">Saved notes remain attached to this event.</span><button class="button" id="saveFeedback" type="submit" ${saves.has(key)?'disabled':''}>${icon('pencil-square')}${saves.has(key)?'Saving…':'Save feedback'}</button></div><p id="feedbackMessage" class="save-message ${message?.error?'error':''}" role="status">${esc(message?.text || '')}</p></form>`)}${section('Saved feedback','Notes for this event, including earlier narrative versions',`<div id="savedFeedback">${saved}</div>`)}</div>`;
}
function bindFeedback(data) {
  const key=draftKey(data), draft=drafts.get(key);
  // Keep this form's controls even if replacing the view triggers a final change.
  const form=$('feedbackForm'), notes=$('feedbackNotes'), subject=$('feedbackSubject'),
    channel=$('feedbackChannel'), reviewer=$('feedbackReviewer'),
    channelField=$('feedbackChannelField'), count=$('feedbackCount');
  subject.value=draft.subject==='narrative'&&!data.narrative?'overall':draft.subject;
  if(draft.channel) channel.value=draft.channel;
  const keep=()=>{
    drafts.set(key,{notes:notes.value,subject:subject.value,channel:subject.value==='channel'?channel.value:'',reviewer:reviewer.value});
    channelField.hidden=subject.value!=='channel';
    count.textContent=notes.value.length.toLocaleString();
    if(D===data && activeTab==='feedback'){
      const tab=$('tab-feedback'), dirty=notes.value.trim().length>0;
      tab.className=dirty?'has-draft':'';
      tab.setAttribute('aria-label',dirty?'Feedback (unsaved draft)':'Feedback');
    }
  };
  form.addEventListener('input',keep);
  form.addEventListener('change',keep);
  form.addEventListener('submit',event=>{event.preventDefault();keep();saveFeedback(data,key);});
}
async function loadFeedback(data) {
  const id=data.event.prediction_id;
  const version=(feedbackVersions.get(id)||0)+1;
  feedbackVersions.set(id,version);
  try { const notes=await getJSON(`/api/event/${id}/feedback`); if(feedbackVersions.get(id)!==version)return; feedbackCache.set(id,{notes}); }
  catch(error) { if(feedbackVersions.get(id)!==version)return; feedbackCache.set(id,{notes:feedbackCache.get(id)?.notes || [],error:error.message}); }
  if(D?.event.prediction_id===id && activeTab==='feedback') $('savedFeedback').innerHTML=savedFeedbackView(feedbackCache.get(id));
}
async function saveFeedback(data,key) {
  if(saves.has(key)) return;
  const draft=drafts.get(key);
  if(!draft.notes.trim()) { feedbackMessages.set(key,{text:'Enter a note before saving.',error:true});render();return; }
  saves.add(key);feedbackMessages.set(key,{text:'Saving feedback…'});render();
  try {
    const note=await getJSON(`/api/event/${data.event.prediction_id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...draft,narrative_id:data.narrative?.narrative_id ?? null})});
    feedbackVersions.set(data.event.prediction_id,(feedbackVersions.get(data.event.prediction_id)||0)+1);
    if(JSON.stringify(drafts.get(key))===JSON.stringify(draft)) drafts.clear(key);
    const existing=feedbackCache.get(data.event.prediction_id)?.notes || [];
    feedbackCache.set(data.event.prediction_id,{notes:[note,...existing.filter(item=>item.feedback_id!==note.feedback_id)]});
    feedbackMessages.set(key,{text:'Feedback saved.'});
    loadFeedback(data);
  } catch(error) { feedbackMessages.set(key,{text:`${error.message} Your draft has been kept.`,error:true}); }
  finally { saves.delete(key); if(D && draftKey(D)===key) render(); }
}
$('patient').addEventListener('change',loadEvents);
$('event').addEventListener('change',event=>loadEvent(event.target.value));
$('refresh').addEventListener('click',()=>{if(!$('patient').value || $('patient').disabled) loadPatients();else if($('event').disabled) loadEvents();else loadEvent($('event').value);});
$('tabs').addEventListener('click',event=>{const button=event.target.closest('[data-tab]');if(button)switchTab(button.dataset.tab,true);});
$('tabs').addEventListener('keydown',event=>{
  if(!D || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();const keys=tabDefinitions().map(([key])=>key), index=keys.indexOf(activeTab);
  const next=event.key==='Home'?0:event.key==='End'?keys.length-1:(index+(event.key==='ArrowRight'?1:-1)+keys.length)%keys.length;
  switchTab(keys[next],true);
});
$('view').addEventListener('click',event=>{const target=event.target.closest('[data-go]');if(target)switchTab(target.dataset.go,true);if(event.target.closest('[data-retry-feedback]') && D)loadFeedback(D);});
loadPatients();


let resizeTimer;
window.addEventListener?.('resize',()=>{
  clearTimeout(resizeTimer);
  resizeTimer=setTimeout(()=>{
    if(!D || activeTab==='feedback')return;
    const states=Array.from(document.querySelectorAll('#view details'),node=>node.open);
    render();
    document.querySelectorAll('#view details').forEach((node,index)=>{if(states[index]!==undefined)node.open=states[index];});
  },150);
});
