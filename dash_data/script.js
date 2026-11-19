  dash.calc('stress_ratio', {
    inputs: ['stress'],
    title: 'Stress test: UnieInfra ÷ vLLM',
    description: 'Each throughput metric of the c32 stress run, UnieInfra optimized divided by vLLM.',
    fn: (stress) => {
      const u = stress.find((r) => r.engine === 'UnieInfra optimized');
      const v = stress.find((r) => r.engine === 'vLLM');
      if (!u || !v) return [];
      return [['rpm', 'Requests/min'], ['input_tps', 'Input tok/s'], ['output_tps', 'Output tok/s']].map(([metric, label]) => ({
        metric, label, unieinfra: Number(u[metric]), vllm: Number(v[metric]),
        ratio: Math.round(Number(u[metric]) / Number(v[metric]) * 100) / 100,
      }));
    },
  });

  const srcRows = (el, id) => {
    const src = dash.data(id);
    if (src.status === 'ok') {
      el.replaceChildren();
      el.classList.remove('dash-skeleton');
      return src.data;
    }
    if (src.status === 'error') failed(el);
    else { el.replaceChildren(); el.classList.add('dash-skeleton'); }
    return null;
  };

  // items: {label, value, color, lo?, hi?, text, tip: [lines], group?}
  function hbars(el, items, opts) {
    const box = d3.select(el);
    const W = Math.max(260, el.clientWidth);
    const rowH = 32;
    const groupH = 24;
    const svg = box.append('svg').attr('width', '100%');
    const m = { t: opts.refs ? 24 : 8, r: 64, b: 28, l: Math.ceil(d3.max(items, (d) => textWidth(svg, d.label))) + 12 };
    let yy = m.t;
    const placed = items.map((d) => {
      if (d.group) yy += groupH;
      const p = { ...d, y: yy };
      yy += rowH;
      return p;
    });
    const Hh = yy + m.b;
    svg.attr('viewBox', `0 0 ${W} ${Hh}`);
    const x = d3.scaleLinear().domain([0, opts.max]).nice().range([m.l, W - m.r]);
    const ticks = x.ticks(W >= 520 ? 6 : 4);
    svg.append('g').selectAll('line').data(ticks).join('line')
      .attr('x1', (t) => x(t)).attr('x2', (t) => x(t)).attr('y1', m.t).attr('y2', Hh - m.b)
      .attr('stroke', 'var(--cds-chart-grid)');
    svg.append('g').selectAll('text').data(ticks).join('text')
      .attr('x', (t) => x(t)).attr('y', Hh - m.b + 16).attr('text-anchor', 'middle')
      .attr('font-size', 12).attr('fill', 'var(--color-fg-muted)').text(opts.fmt);
    for (const ref of opts.refs || []) {
      svg.append('line').attr('x1', x(ref.value)).attr('x2', x(ref.value)).attr('y1', m.t - 4).attr('y2', Hh - m.b)
        .attr('stroke', 'var(--color-fg-muted)').attr('stroke-width', 1.5)
        .attr('stroke-dasharray', ref.dashed ? '4 3' : null);
      svg.append('text').attr('x', x(ref.value)).attr('y', m.t - 8).attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text(ref.label);
    }
    const tip = tipOf(box);
    for (const d of placed) {
      if (d.group) {
        svg.append('text').attr('x', 0).attr('y', d.y - 8).attr('font-size', 12)
          .attr('font-weight', 600).attr('fill', 'var(--color-fg)').text(d.group);
      }
      const cy = d.y + rowH / 2;
      svg.append('text').attr('x', m.l - 8).attr('y', cy).attr('dy', '0.32em').attr('text-anchor', 'end')
        .attr('font-size', 12).attr('fill', 'var(--color-fg-muted)').text(d.label);
      svg.append('rect').attr('x', m.l).attr('y', d.y + 6).attr('height', rowH - 12)
        .attr('width', Math.max(1, x(d.value) - m.l)).attr('rx', 4).attr('fill', d.color)
        .on('mousemove', (ev) => {
          tip.style('display', null).html('');
          tip.append('div').attr('class', 'mx-tip-h').text(d.tipHead || d.label);
          for (const line of d.tip) tip.append('div').text(line);
          placeTip(tip, box, ev);
        })
        .on('mouseleave', () => tip.style('display', 'none'));
      let end = x(d.value);
      if (d.lo != null && d.hi != null) {
        const g = svg.append('g').attr('stroke', 'var(--color-fg)').attr('stroke-width', 1.5).attr('pointer-events', 'none');
        g.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi)).attr('y1', cy).attr('y2', cy);
        g.append('line').attr('x1', x(d.lo)).attr('x2', x(d.lo)).attr('y1', cy - 5).attr('y2', cy + 5);
        g.append('line').attr('x1', x(d.hi)).attr('x2', x(d.hi)).attr('y1', cy - 5).attr('y2', cy + 5);
        end = Math.max(end, x(d.hi));
      }
      svg.append('text').attr('x', end + 6).attr('y', cy).attr('dy', '0.32em')
        .attr('font-size', 12).attr('fill', 'var(--color-fg)').text(d.text);
    }
  }

  dash.draw('stress-ratio', (el) => {
    const rows = srcRows(el, 'stress_ratio');
    if (!rows) return;
    const big = (v) => (v >= 10000 ? compact(v) : d3.format(',.1f')(v));
    hbars(el, rows.map((r) => ({
      label: r.label, value: r.ratio, color: dash.colors[0], text: `${d3.format('.2f')(r.ratio)}×`,
      tip: [`UnieInfra ${big(r.unieinfra)}`, `vLLM ${big(r.vllm)}`],
    })), {
      max: Math.max(2.2, d3.max(rows, (r) => r.ratio) * 1.1),
      fmt: (t) => `${t}×`,
      refs: [{ value: 1, label: 'vLLM' }, { value: 2, label: '2× goal', dashed: true }],
    });
  });

  dash.draw('stress-lat', (el) => {
    const rows = srcRows(el, 'stress');
    if (!rows) return;
    const order = ['UnieInfra optimized', 'vLLM'];
    const items = [];
    for (const [field, title] of [['ttft_p50_s', 'TTFT p50'], ['latency_p50_s', 'Request latency p50']]) {
      order.forEach((e, i) => {
        const r = rows.find((q) => q.engine === e);
        if (!r) return;
        const v = Number(r[field]);
        items.push({
          group: i === 0 ? title : null, label: e, value: v, color: color(e),
          text: `${d3.format(',.2~f')(v)} s`, tipHead: `${title}, ${e}`,
          tip: [`${d3.format(',.2~f')(v)} s`, `${fmt0(r.requests)} requests in ${fmt0(r.duration_s)} s`],
        });
      });
    }
    hbars(el, items, { max: d3.max(items, (d) => d.value) * 1.08, fmt: (t) => `${t} s` });
  });

  dash.draw('prefill', (el) => {
    const rows = srcRows(el, 'prefill_ab');
    if (!rows) return;
    hbars(el, rows.map((r) => ({
      label: r.label, value: Number(r.mean_s), lo: Number(r.best_s), hi: Number(r.worst_s),
      color: r.variant === 'iris' ? dash.colors[0] : 'var(--cds-chart-muted)',
      text: `${d3.format('.2f')(r.mean_s)} s`,
      tip: [`mean ${d3.format('.2f')(r.mean_s)} s`, `range ${r.best_s}–${r.worst_s} s`, r.runs ? `${r.runs} runs` : 'range from the log'],
    })), { max: d3.max(rows, (r) => Number(r.worst_s)) * 1.1, fmt: (t) => `${t} s` });
  });

