document.addEventListener('DOMContentLoaded', () => {

    // ── 1. DOM Elements References ────────────────────────────
    const runList = document.getElementById('runList');
    const runListState = document.getElementById('runListState');
    const goOcrBtn = document.getElementById('goOcrBtn');

    const addRunCard = document.getElementById('addRunCard');
    const addRunForm = document.getElementById('addRunForm');
    const addRunError = document.getElementById('addRunError');
    const showAddRunBtn = document.getElementById('showAddRunBtn');
    const cancelAddRunBtn = document.getElementById('cancelAddRunBtn');
    const saveRunBtn = document.getElementById('saveRunBtn');

    // ── 2. State ──────────────────────────────────────────────
    let selectedRunId = null;

    // จำ Run ที่เลือกล่าสุดไว้ preselect ครั้งหน้า (สะดวกเฉยๆ ไม่มีก็ไม่เป็นไร)
    function loadLastRunId() {
        try { return localStorage.getItem('lastRunId'); } catch (e) { return null; }
    }
    function saveLastRunId(id) {
        try { localStorage.setItem('lastRunId', String(id)); } catch (e) { /* ignore */ }
    }
    function clearLastRunId() {
        try { localStorage.removeItem('lastRunId'); } catch (e) { /* ignore */ }
    }

    // ── 3. Load & Render Run List ─────────────────────────────
    async function loadRuns(preselectId) {
        runListState.textContent = 'กำลังโหลด…';
        runListState.classList.remove('hidden', 'error');
        try {
            const res = await fetch('/api/meter-runs');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const runs = await res.json();
            renderRuns(runs, preselectId ?? loadLastRunId());
        } catch (err) {
            console.error(err);
            runList.innerHTML = '';
            runListState.textContent = 'โหลดรายการ Meter Run ไม่สำเร็จ — ตรวจสอบว่า server ทำงานอยู่';
            runListState.classList.add('error');
            selectRun(null);
        }
    }

    function renderRuns(runs, preselectId) {
        runList.innerHTML = '';
        if (runs.length === 0) {
            runListState.textContent = 'ยังไม่มี Meter Run ในระบบ — กด "เพิ่ม Meter Run" เพื่อเริ่มต้น';
            selectRun(null);
            return;
        }
        runListState.classList.add('hidden');

        runs.forEach(run => runList.appendChild(buildRunItem(run)));

        const exists = runs.some(r => String(r.id) === String(preselectId));
        selectRun(exists ? Number(preselectId) : null);
    }

    // การ์ด 1 Run: แสดง FY และ FE คู่กันเป็นชุด (DOM ปลอดภัย: textContent เท่านั้น)
    function buildRunItem(run) {
        const item = document.createElement('label');
        item.className = 'run-item';
        item.dataset.runId = run.id;

        const radio = document.createElement('input');
        radio.type = 'radio';
        radio.name = 'meterRun';
        radio.value = run.id;
        radio.addEventListener('change', () => selectRun(run.id));

        const body = document.createElement('div');
        body.className = 'run-item-body';

        const title = document.createElement('div');
        title.className = 'run-item-title';
        title.textContent = `${run.company} · Run ${run.run_label}`;
        body.appendChild(title);

        body.appendChild(buildDeviceRow('Electronic Volume Corrector', 'FY', run.tag_fy, run.serial_fy));
        body.appendChild(buildDeviceRow('Turbine Meter', 'FE', run.tag_fe, run.serial_fe));

        item.appendChild(radio);
        item.appendChild(body);
        item.appendChild(buildDeleteControls(run, body));

        // ดับเบิลคลิก = เลือกแล้วไปหน้า OCR เลย
        item.addEventListener('dblclick', () => {
            selectRun(run.id);
            goToOcr();
        });
        return item;
    }

    // ── ปุ่มลบ: Run ที่มีประวัติการจดแล้วลบไม่ได้ (server บล็อกซ้ำอีกชั้น) ──
    // ยืนยันในการ์ด (ไม่ใช้ confirm() ของ browser)
    function buildDeleteControls(run, body) {
        const wrap = document.createElement('div');
        wrap.className = 'run-item-actions';

        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'icon-btn danger';
        delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        const inUse = run.reading_count > 0;
        delBtn.disabled = inUse;
        delBtn.title = inUse
            ? `ลบไม่ได้ — มีประวัติการจด ${run.reading_count} รายการ`
            : 'ลบ Meter Run นี้';
        delBtn.setAttribute('aria-label', delBtn.title);
        wrap.appendChild(delBtn);

        // แถบยืนยัน (ซ่อนจนกดถังขยะ)
        const confirmBar = document.createElement('div');
        confirmBar.className = 'run-delete-confirm hidden';
        const msg = document.createElement('span');
        msg.textContent = `ลบ ${run.company} · Run ${run.run_label} ?`;
        const yes = document.createElement('button');
        yes.type = 'button';
        yes.className = 'btn btn-sm btn-danger';
        yes.innerHTML = '<i class="fa-solid fa-trash-can"></i> ยืนยันลบ';
        const no = document.createElement('button');
        no.type = 'button';
        no.className = 'btn btn-sm btn-outline';
        no.textContent = 'ยกเลิก';
        const err = document.createElement('div');
        err.className = 'run-delete-error hidden';
        confirmBar.append(msg, yes, no, err);
        body.appendChild(confirmBar);

        // ปุ่มอยู่ใน <label> -> กันคลิกไปเลือก Run / ดับเบิลคลิกไปหน้า OCR
        [delBtn, confirmBar].forEach(node => {
            node.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); });
            node.addEventListener('dblclick', e => { e.preventDefault(); e.stopPropagation(); });
        });

        delBtn.addEventListener('click', () => {
            confirmBar.classList.remove('hidden');
            err.classList.add('hidden');
            delBtn.disabled = true;
        });
        no.addEventListener('click', () => {
            confirmBar.classList.add('hidden');
            delBtn.disabled = inUse;
        });
        yes.addEventListener('click', async () => {
            yes.disabled = true;
            no.disabled = true;
            try {
                const res = await fetch(`/api/meter-runs/${encodeURIComponent(run.id)}`, { method: 'DELETE' });
                if (!res.ok) {
                    const data = await res.json().catch(() => ({}));
                    throw new Error(typeof data.detail === 'string' ? data.detail : 'ลบไม่สำเร็จ');
                }
                if (String(loadLastRunId()) === String(run.id)) clearLastRunId();
                await loadRuns(selectedRunId === run.id ? null : selectedRunId);
            } catch (e) {
                console.error(e);
                err.textContent = e.message || 'เกิดข้อผิดพลาดในการติดต่อเซิร์ฟเวอร์';
                err.classList.remove('hidden');
                yes.disabled = false;
                no.disabled = false;
            }
        });
        return wrap;
    }

    function buildDeviceRow(deviceName, code, tag, serial) {
        const row = document.createElement('div');
        row.className = 'run-device';

        const name = document.createElement('span');
        name.className = 'run-device-name';
        name.textContent = `${deviceName} (${code})`;

        const tagEl = document.createElement('span');
        tagEl.className = 'run-device-tag';
        tagEl.textContent = tag || '—';

        const serialEl = document.createElement('span');
        serialEl.className = 'run-device-serial';
        serialEl.textContent = serial ? `S/N ${serial}` : 'S/N —';

        row.append(name, tagEl, serialEl);
        return row;
    }

    // ── 4. Selection (ทีละ 1 Run) ─────────────────────────────
    function selectRun(id) {
        selectedRunId = id;
        runList.querySelectorAll('.run-item').forEach(item => {
            const isSel = id !== null && Number(item.dataset.runId) === id;
            item.classList.toggle('selected', isSel);
            item.querySelector('input').checked = isSel;
        });
        goOcrBtn.disabled = selectedRunId === null;
    }

    function goToOcr() {
        if (selectedRunId === null) return;
        saveLastRunId(selectedRunId);
        window.location.href = `/ocr?run=${encodeURIComponent(selectedRunId)}`;
    }

    goOcrBtn.addEventListener('click', goToOcr);

    // ── 5. Add Meter Run Form ─────────────────────────────────
    function showForm(show) {
        addRunCard.classList.toggle('hidden', !show);
        showAddRunBtn.disabled = show;
        hideError();
        if (show) {
            addRunForm.reset();
            addRunForm.elements.company.focus();
        }
    }

    function showError(msg) {
        addRunError.textContent = msg;
        addRunError.classList.remove('hidden');
    }
    function hideError() {
        addRunError.textContent = '';
        addRunError.classList.add('hidden');
    }

    showAddRunBtn.addEventListener('click', () => showForm(true));
    cancelAddRunBtn.addEventListener('click', () => showForm(false));

    addRunForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();

        const payload = {};
        ['company', 'run_label', 'tag_fy', 'serial_fy', 'tag_fe', 'serial_fe'].forEach(k => {
            payload[k] = addRunForm.elements[k].value.trim();
        });
        if (!payload.company || !payload.run_label) {
            showError('กรุณากรอกบริษัทและ Run');
            return;
        }

        saveRunBtn.disabled = true;
        try {
            const res = await fetch('/api/meter-runs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) {
                // 409 = Run ซ้ำ -> server ส่งข้อความที่คนอ่านเข้าใจมาให้แล้ว
                const detail = typeof data.detail === 'string' ? data.detail : 'บันทึกไม่สำเร็จ';
                throw new Error(detail);
            }
            showForm(false);
            await loadRuns(data.id);  // เลือก Run ที่เพิ่งเพิ่มให้อัตโนมัติ
        } catch (err) {
            console.error(err);
            showError(err.message || 'เกิดข้อผิดพลาดในการติดต่อเซิร์ฟเวอร์');
        } finally {
            saveRunBtn.disabled = false;
        }
    });

    loadRuns();
});
