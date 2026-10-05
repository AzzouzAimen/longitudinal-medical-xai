/* Pure helpers, also tested in Node without a browser. */
(function(root, factory) {
  const core = factory();
  if (typeof module === 'object' && module.exports) module.exports = core;
  else root.ViewerCore = core;
})(globalThis, function() {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;'}[char]));
  const finite = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const format = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('en', {minimumFractionDigits:digits, maximumFractionDigits:digits}) : 'No data';
  const time = value => {
    const date = new Date(value);
    if (!value || !Number.isFinite(date.getTime())) return 'No data';
    const fraction = date.getUTCMilliseconds() ? '.' + String(date.getUTCMilliseconds()).padStart(3,'0').replace(/0+$/,'') : '';
    return date.toISOString().slice(11,19) + fraction;
  };
  const dateTime = value => {
    const date = new Date(value);
    return value && Number.isFinite(date.getTime()) ? `${date.toISOString().slice(0,10)} ${time(value)} UTC` : 'No data';
  };
  class RequestGate {
    constructor() { this.version = 0; }
    begin() { return ++this.version; }
    current(version) { return version === this.version; }
  }
  class Drafts {
    constructor(storage) { this.storage = storage; this.memory = new Map(); }
    get(key) {
      if (this.memory.has(key)) return {...this.memory.get(key)};
      try {
        const saved = JSON.parse(this.storage?.getItem(`pdm-feedback:${key}`) || 'null');
        if (saved && typeof saved.notes === 'string') { this.memory.set(key,saved); return {...saved}; }
      } catch (_) {}
      return {notes:'', subject:'overall', channel:'', reviewer:''};
    }
    set(key,value) {
      this.memory.set(key,{...value});
      try { this.storage?.setItem(`pdm-feedback:${key}`,JSON.stringify(value)); } catch (_) {}
    }
    clear(key) {
      this.memory.delete(key);
      try { this.storage?.removeItem(`pdm-feedback:${key}`); } catch (_) {}
    }
  }
  function signalPlot(points,event,options={}) {
    const start = new Date(event.start_time).getTime(), duration = Number(event.duration_seconds);
    const data = (points || []).filter(p => finite(p.value) && Number.isFinite(new Date(p.timestamp).getTime()));
    if (!data.length || !Number.isFinite(start) || !(duration > 0)) return '<div class="empty plot-empty">No signal data stored for this window.</div>';
    const W=finite(options.width)?Math.max(320,Math.min(720,Number(options.width))):720,H=options.compact?115:200,left=64,right=W-18,top=22,bottom=H-30;
    let lo=Math.min(...data.map(p=>Number(p.value))),hi=Math.max(...data.map(p=>Number(p.value)));
    const margin=(hi-lo)*.08 || Math.abs(hi)*.08 || 1; lo-=margin; hi+=margin;
    const X=t=>Math.max(left,Math.min(right,left+(new Date(t).getTime()-start)/(duration*1000)*(right-left)));
    const Y=v=>bottom-(Number(v)-lo)/(hi-lo)*(bottom-top);
    const tick=v=>Math.abs(v)>0 && (Math.abs(v)<.01 || Math.abs(v)>=10000)?v.toExponential(1):Number(v.toPrecision(3)).toString();
    const path=data.map(p=>`${X(p.timestamp).toFixed(2)},${Y(p.value).toFixed(2)}`).join(' ');
    const markers=(options.markers||[]).filter(m=>m.time && Number.isFinite(new Date(m.time).getTime())).map(m=>`<line class="plot-marker" x1="${X(m.time)}" x2="${X(m.time)}" y1="${top}" y2="${bottom}"><title>${esc(m.label)}</title></line>`).join('');
    const regions=(options.regions||[]).filter(r=>r.temporal_start && r.temporal_end && new Date(r.temporal_end)>new Date(r.temporal_start)).map(r=>`<rect class="plot-region" x="${X(r.temporal_start)}" y="${top}" width="${Math.max(0,X(r.temporal_end)-X(r.temporal_start))}" height="${bottom-top}"><title>Stored residual region</title></rect>`).join('');
    return `<svg class="plot ${options.attention?'attention-plot':''}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(options.label||'Recorded signal')} (${esc(options.unit||'unit unavailable')})"><title>${esc(options.label||'Recorded signal')} · ${esc(options.unit||'unit unavailable')}</title>
      ${[lo,(lo+hi)/2,hi].map(v=>`<line class="plot-grid" x1="${left}" x2="${right}" y1="${Y(v)}" y2="${Y(v)}"/><text class="plot-tick" x="${left-8}" y="${Y(v)+4}" text-anchor="end">${tick(v)}</text>`).join('')}
      ${regions}<polyline class="plot-trace" points="${path}"/>${markers}
      <text class="plot-tick" x="${left}" y="${H-8}">0 s</text><text class="plot-tick" x="${right}" y="${H-8}" text-anchor="end">${esc(duration)} s</text><text class="plot-tick" x="${left}" y="13">${esc(options.unit||'Unit unavailable')}</text></svg>`;
  }
  return {esc,finite,format,time,dateTime,RequestGate,Drafts,signalPlot};
});
