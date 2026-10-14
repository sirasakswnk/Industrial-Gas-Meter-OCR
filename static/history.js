document.addEventListener('DOMContentLoaded', () => {

    // ── 1. DOM Elements References ────────────────────────────
    const runFilter = document.getElementById('runFilter');
    const historyState = document.getElementById('historyState');
    const historyTable = document.getElementById('historyTable');
    const historyBody = document.getElementById('historyBody');
    const historySummary = document.getElementById('historySummary');

    // ── 2. State ──────────────────────────────────────────────
    let fieldDefs = [];
    let turbineKey = 'turbine_v_m3';
    let openReadingId = null;   // แถวที่เปิดรายละเอียดอยู่ (ทีละแถว)
    const COLUMN_COUNT = 7;

    // ── 3. Init: ตารางฟิลด์ + รายการ Run + ประวัติ ─────────────
    async function init() {
        try {
            const [fieldsRes, runsRes] = await Promise.all([
                fetch('/api/reading-fields'), fetch('/api/meter-runs'),
            ]);
            const fieldsData = await fieldsRes.json();
            fieldDefs = fieldsData.fields;
            turbineKey = fieldsData.turbine_field;
            const runs = await runsRes.json();
            runs.forEach(run => {
                const opt = document.createElement('option');
                opt.value = run.id;
                opt.textContent = `${run.company} · Run ${run.run_label}`;
                runFilter.appendChild(opt);
            });

            // ?run=<id> -> เลือกไว้ให้ (มาจากปุ่ม "ประวัติ" บนหน้า OCR)
            const param = new URLSearchParams(window.location.search).get('run');
            if (param && runs.some(r => String(r.id) === param)) runFilter.value = param;
        } catch (err) {
            console.error(err);
            showState('โหลดข้อมูลไม่สำเร็จ — ตรวจสอบว่า server ทำงานอยู่', true);
            return;
        }
        loadReadings();
    }

    runFilter.addEventListener('change', () => {
        const url = new URL(window.location.href);
        if (runFilter.value) url.searchParams.set('run', runFilter.value);
        else url.searchParams.delete('run');
        window.history.replaceState(null, '', url);
        loadReadings();
    });

    function showState(text, isError = false) {
        historyState.textContent = text;
        historyState.classList.toggle('error', isError);
        historyState.classList.remove('hidden');
        historyTable.classList.add('hidden');
    }

    // ── 4. รายการประวัติ ──────────────────────────────────────
    async function loadReadings() {
        showState('กำลังโหลด…');
        historySummary.textContent = 'กำลังโหลด…';
        openReadingId = null;
        const query = runFilter.value ? `?meter_run_id=${encodeURIComponent(runFilter.value)}` : '';
        try {
            const res = await fetch(`/api/readings${query}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            renderReadings(await res.json());
        } catch (err) {
            console.error(err);
            showState('โหลดประวัติไม่สำเร็จ', true);
            historySummary.textContent = '';
        }
    }

    // ค่าที่มีในรายการ = ฟิลด์ที่โมเดลอ่านได้ หรือคนกรอกเอง (ใช้เป็นตัวหารของ "แก้ด้วยมือ")
    function presentFields(r) {
        return fieldDefs.filter(f =>
            (r.raw_values || {})[f.key] != null || (r.final_values || {})[f.key] != null);
    }

    function renderReadings(readings) {
        historyBody.innerHTML = '';
        renderSummary(readings);
        if (readings.length === 0) {
            showState(runFilter.value ? 'ยังไม่มีการจดค่าของ Meter Run นี้' : 'ยังไม่มีการจดค่าในระบบ');
            return;
        }
        historyState.classList.add('hidden');
        historyTable.classList.remove('hidden');
        readings.forEach(r => historyBody.appendChild(buildRow(r)));
    }

    // สรุปอัตราค่าที่ไม่ถูกแก้ด้วยมือ (ไม่ใช่ความแม่นยำจริง — ดู tooltip)
    function renderSummary(readings) {
        const total = readings.reduce((n, r) => n + presentFields(r).length, 0);
        const edited = readings.reduce((n, r) => n + (r.edited_fields || []).length, 0);
        const withEdit = readings.filter(r => (r.edited_fields || []).length > 0).length;
        if (readings.length === 0) {
            historySummary.textContent = '0 รายการ';
            return;
        }
        const pct = total > 0 ? ((total - edited) / total * 100).toFixed(1) : '—';
        historySummary.textContent =
            `${readings.length} รายการ · มีการแก้ด้วยมือ ${withEdit} รายการ · ` +
            `อัตราค่าที่ไม่ถูกแก้ด้วยมือ ${total - edited}/${total} (${pct}%)`;
        // ไม่ใช่ความแม่นยำจริง: ถ้าผู้ใช้กดบันทึกโดยไม่ตรวจ ตัวเลขจะขึ้น 100% ทั้งที่อาจผิด
        historySummary.title = 'นับจากค่าที่ผู้ใช้ไม่ได้แก้ ไม่ใช่การวัดความถูกต้องเทียบกับค่าจริงบนมิเตอร์ ' +
            '(นับเฉพาะค่าที่มีในรายการ — การติ๊กตัดทศนิยม Turbine ไม่นับว่าเป็นการแก้)';
    }

    function buildRow(r) {
        const tr = document.createElement('tr');
        tr.className = 'history-row';
        tr.dataset.readingId = r.id;

        tr.appendChild(td(formatDateTime(r.recorded_at)));
        tr.appendChild(td(r.run ? `${r.run.company} · Run ${r.run.run_label}` : `#${r.meter_run_id}`));
        ['correct_v_scm', 'line_v_m3', turbineKey].forEach(key => {
            const v = (r.final_values || {})[key];
            tr.appendChild(td(v ?? '—', 'align-right num' + (v == null ? ' val-null' : '')));
        });

        // จำนวนฟิลด์ที่ถูกแก้ด้วยมือ / ค่าที่มี — hover ดูชื่อฟิลด์
        const edited = r.edited_fields || [];
        const editCell = td('', 'align-center');
        const pill = document.createElement('span');
        pill.className = `edit-count ${edited.length ? 'has-edit' : ''}`;
        pill.textContent = `${edited.length}/${presentFields(r).length}`;
        pill.title = edited.length
            ? 'แก้: ' + edited.map(labelOf).join(', ')
            : 'ไม่มีการแก้ด้วยมือ';
        editCell.appendChild(pill);
        if ((r.low_confidence_fields || []).length) {
            const warn = document.createElement('i');
            warn.className = 'fa-solid fa-triangle-exclamation low-conf-icon';
            warn.title = 'โมเดลไม่มั่นใจ: ' + r.low_confidence_fields.map(labelOf).join(', ');
            editCell.appendChild(warn);
        }
        tr.appendChild(editCell);

        // ปุ่ม: รายละเอียด + PDF
        const actions = td('', 'align-right history-actions');
        const detailBtn = document.createElement('button');
        detailBtn.type = 'button';
        detailBtn.className = 'btn btn-outline btn-sm';
        detailBtn.innerHTML = '<i class="fa-solid fa-chevron-down"></i> รายละเอียด';
        detailBtn.addEventListener('click', (e) => { e.stopPropagation(); toggleDetail(r.id, tr); });
        actions.appendChild(detailBtn);
        actions.appendChild(pdfLink(r.id, true));
        tr.appendChild(actions);

        tr.addEventListener('click', () => toggleDetail(r.id, tr));
        return tr;
    }

    function pdfLink(id, small) {
        const a = document.createElement('a');
        a.href = `/api/readings/${id}/pdf`;
        a.className = `btn btn-outline ${small ? 'btn-sm' : ''}`;
        a.innerHTML = '<i class="fa-solid fa-file-pdf"></i> PDF';
        a.title = 'Export PDF อีกครั้ง';
        a.addEventListener('click', e => e.stopPropagation());
        return a;
    }

    // ── 5. รายละเอียด (เปิดใต้แถว ทีละรายการ) ──────────────────
    async function toggleDetail(id, tr) {
        const existing = historyBody.querySelector('.history-detail-row');
        if (existing) existing.remove();
        historyBody.querySelectorAll('.history-row.open').forEach(row => row.classList.remove('open'));
        if (openReadingId === id) {
            openReadingId = null;
            return;
        }
        openReadingId = id;
        tr.classList.add('open');

        const detailRow = document.createElement('tr');
        detailRow.className = 'history-detail-row';
        const cell = document.createElement('td');
        cell.colSpan = COLUMN_COUNT;
        cell.textContent = 'กำลังโหลด…';
        detailRow.appendChild(cell);
        tr.after(detailRow);

        try {
            const res = await fetch(`/api/readings/${id}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (openReadingId !== id) return;  // ผู้ใช้กดแถวอื่นไปแล้ว
            cell.textContent = '';
            cell.appendChild(buildDetail(data));
        } catch (err) {
            console.error(err);
            cell.textContent = 'โหลดรายละเอียดไม่สำเร็จ';
        }
    }

    function buildDetail({ reading, run, previous }) {
        const wrap = el('div', 'history-detail');

        // ข้อมูลทั่วไป
        const meta = el('div', 'detail-meta');
        const metaItems = [
            ['Meter Run', run ? `${run.company} · Run ${run.run_label}` : '—'],
            ['Tag FY / FE', run ? `${run.tag_fy || '—'} / ${run.tag_fe || '—'}` : '—'],
            ['วัน/เวลาที่จด', formatDateTime(reading.recorded_at)],
            ['บันทึกในระบบเมื่อ', formatDateTime(reading.created_at)],
            ['ค่าจดครั้งก่อน', previous ? formatDateTime(previous.recorded_at) : 'ไม่มี (ครั้งแรกของ Run)'],
            ['ตัดทศนิยม Turbine', reading.turbine_trim_digits ? `${reading.turbine_trim_digits} หลัก` : 'ไม่ตัด'],
            ['Engine', reading.engine || '—'],
        ];
        metaItems.forEach(([k, v]) => {
            const item = el('div', 'detail-meta-item');
            item.append(el('span', 'detail-meta-label', k), el('span', 'detail-meta-value', v));
            meta.appendChild(item);
        });
        wrap.appendChild(meta);

        // ตาราง ค่าที่โมเดลอ่าน vs ค่าที่บันทึก — ใช้ดูว่าโมเดลอ่านผิดตรงไหน
        const edited = new Set(reading.edited_fields || []);
        const lowConf = new Set(reading.low_confidence_fields || []);
        const table = el('table', 'params-table detail-table');
        const thead = el('thead');
        const headRow = el('tr');
        ['รายการ', 'หน่วย', 'ค่าที่โมเดลอ่าน', 'ค่าที่บันทึก', 'สถานะ'].forEach((h, i) => {
            headRow.appendChild(el('th', i >= 2 && i <= 3 ? 'align-right' : '', h));
        });
        thead.appendChild(headRow);
        table.appendChild(thead);
        const tbody = el('tbody');
        fieldDefs.forEach(f => {
            const raw = (reading.raw_values || {})[f.key];
            const final = (reading.final_values || {})[f.key];
            const row = el('tr');
            row.appendChild(el('td', 'param-name', f.label));
            row.appendChild(el('td', '', f.unit));
            row.appendChild(el('td', 'align-right num' + (raw == null ? ' val-null' : ''), raw ?? '—'));
            row.appendChild(el('td', 'align-right num' + (final == null ? ' val-null' : ''), final ?? '—'));
            const status = el('td');
            if (edited.has(f.key)) {
                status.appendChild(badge('edited', 'fa-pen', 'แก้ด้วยมือ'));
            }
            if (lowConf.has(f.key)) {
                status.appendChild(badge('warn', 'fa-triangle-exclamation', 'โมเดลไม่มั่นใจ'));
            }
            if (f.key === turbineKey && reading.turbine_trim_digits && final != null) {
                status.appendChild(badge('null', 'fa-scissors', `ตัด ${reading.turbine_trim_digits} หลัก`));
            }
            row.appendChild(status);
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        const tableWrap = el('div', 'detail-table-wrap');
        tableWrap.appendChild(table);
        wrap.appendChild(tableWrap);

        // ภาพที่ใช้อ่านค่า
        const images = reading.image_paths || [];
        if (images.length) {
            const gallery = el('div', 'detail-images');
            images.forEach((path, i) => {
                const a = document.createElement('a');
                a.href = `/api/readings/${reading.id}/images/${i}`;
                a.target = '_blank';
                a.rel = 'noopener';
                a.title = path.split('/').pop();
                const img = document.createElement('img');
                img.src = a.href;
                img.alt = `ภาพที่ ${i + 1}`;
                img.loading = 'lazy';
                img.addEventListener('error', () => {
                    a.replaceWith(el('span', 'detail-image-missing', `ภาพที่ ${i + 1} ไม่พบไฟล์`));
                });
                a.appendChild(img);
                gallery.appendChild(a);
            });
            wrap.appendChild(gallery);
        }

        const actions = el('div', 'detail-actions');
        actions.appendChild(pdfLink(reading.id, false));
        wrap.appendChild(actions);
        return wrap;
    }

    // ── 6. Utilities ──────────────────────────────────────────
    function labelOf(key) {
        const f = fieldDefs.find(d => d.key === key);
        return f ? f.label : key;
    }

    function td(text, className) {
        return el('td', className, text);
    }

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function badge(kind, icon, text) {
        const b = el('span', `status-badge ${kind}`);
        const i = document.createElement('i');
        i.className = `fa-solid ${icon}`;
        b.append(i, ` ${text}`);
        return b;
    }

    function formatDateTime(iso) {
        if (!iso) return '—';
        const [date, time = ''] = iso.split('T');
        const [y, m, d] = date.split('-');
        return `${d}/${m}/${y} ${time.slice(0, 5)}`;
    }

    init();
});
