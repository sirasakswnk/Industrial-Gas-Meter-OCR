document.addEventListener('DOMContentLoaded', () => {

    // ── 1. Meter Run ที่เลือกจากหน้าแรก (theme toggle ย้ายไป theme.js) ──
    // หน้านี้ต้องมี ?run=<id> เสมอ — ไม่มี/ไม่พบในระบบ -> เด้งกลับหน้าแรกให้เลือกก่อน
    let currentRun = null;
    const runId = new URLSearchParams(window.location.search).get('run');

    async function loadCurrentRun() {
        if (!runId || !/^\d+$/.test(runId)) {
            window.location.replace('/');
            return;
        }
        try {
            const res = await fetch(`/api/meter-runs/${runId}`);
            if (!res.ok) {
                window.location.replace('/');
                return;
            }
            currentRun = await res.json();
            renderRunBar(currentRun);
            loadPrevReading();  // ค่าจดครั้งก่อนของ Run นี้ (ส่วนที่ 9)
        } catch (err) {
            console.error(err);
            document.getElementById('runBarTitle').textContent = 'โหลดข้อมูล Meter Run ไม่สำเร็จ';
        }
    }

    function renderRunBar(run) {
        document.getElementById('runBarTitle').textContent = `${run.company} · Run ${run.run_label}`;
        document.getElementById('runBarFy').textContent = run.tag_fy || '—';
        document.getElementById('runBarFe').textContent = run.tag_fe || '—';
        document.getElementById('runHistoryLink').href = `/history?run=${encodeURIComponent(run.id)}`;
    }

    loadCurrentRun();

    // ── 2. DOM Elements References ────────────────────────────
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreviewGrid = document.getElementById('imagePreviewGrid');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const processBtn = document.getElementById('processBtn');

    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsError = document.getElementById('resultsError');
    const errorText = document.getElementById('errorText');
    const retryBtn = document.getElementById('retryBtn');
    const resultsTableContainer = document.getElementById('resultsTableContainer');
    const resultsJsonContainer = document.getElementById('resultsJsonContainer');
    const jsonPreCode = document.getElementById('jsonPreCode');
    const copyJsonBtn = document.getElementById('copyJsonBtn');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const chipBtns = document.querySelectorAll('.chip');

    // ── 3. System States ─────────────────────────────────────
    let selectedFiles = [];
    let isSampleFile = false;
    let sampleFileName = '';
    let currentTab = 'table-view';
    let currentResult = null;

    // ── 4. UI State Helpers ───────────────────────────────────
    function showOnly(...toShow) {
        const allStates = [
            resultsPlaceholder, resultsLoading, resultsError,
            resultsTableContainer, resultsJsonContainer
        ];
        allStates.forEach(el => el.classList.add('hidden'));
        toShow.forEach(el => el && el.classList.remove('hidden'));
    }

    function setActiveTab(tab) {
        currentTab = tab;
        tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
        if (currentResult) {
            showOnly(tab === 'table-view' ? resultsTableContainer : resultsJsonContainer);
        }
    }

    tabBtns.forEach(btn => btn.addEventListener('click', () => setActiveTab(btn.dataset.tab)));

    // ── 5. Dropzone ───────────────────────────────────────────
    dropzone.addEventListener('click', (e) => {
        if (e.target === removeImageBtn || removeImageBtn.contains(e.target)) return;
        if (e.target.closest('.remove-one-btn')) return;
        fileInput.click();
    });

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFilesSelect(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFilesSelect(e.target.files);
    });

    // ── 6. File Handling ──────────────────────────────────────

    // ลายนิ้วมือของไฟล์สำหรับ dedupe: ชื่อ + ขนาด + lastModified
    function fileKey(f) {
        return `${f.name}|${f.size}|${f.lastModified}`;
    }

    function handleFilesSelect(files) {
        const filesArray = Array.from(files);
        const validImages = filesArray.filter(
            f => f.type.startsWith('image/') || f.name.match(/\.(jpg|jpeg|png|gif)$/i)
        );
        if (validImages.length === 0) {
            alert('กรุณาเลือกไฟล์รูปภาพที่เป็นมาตรฐาน (PNG, JPG, JPEG) เท่านั้นครับ');
            return;
        }

        // ── Dedupe: ข้ามไฟล์ที่มีอยู่แล้ว ──
        const existingKeys = new Set(selectedFiles.map(fileKey));
        const newFiles = validImages.filter(f => !existingKeys.has(fileKey(f)));
        if (newFiles.length === 0 && selectedFiles.length > 0) return;

        selectedFiles = [...selectedFiles, ...newFiles];
        isSampleFile = false;
        sampleFileName = '';
        chipBtns.forEach(c => c.classList.remove('active'));
        renderPreviews();
    }

    function renderPreviews() {
        // Revoke URL เก่ากันหน่วยความจำรั่ว
        imagePreviewGrid.querySelectorAll('img[data-objurl]').forEach(img => {
            URL.revokeObjectURL(img.src);
        });
        imagePreviewGrid.innerHTML = '';

        // ── พรีวิวเรียงแถวเดียว (1fr) ให้การ์ดกว้างพอสำหรับปุ่ม X ──
        imagePreviewGrid.style.gridTemplateColumns = '1fr';

        selectedFiles.forEach((file, index) => {
            const imgCard = document.createElement('div');
            imgCard.className = 'chip active';
            imgCard.style.cssText = 'display:flex; align-items:center; padding:6px 8px; cursor:default; gap:8px; min-width:0;';

            const thumb = document.createElement('img');
            thumb.src = URL.createObjectURL(file);
            thumb.dataset.objurl = '1';
            thumb.style.cssText = 'width:40px; height:40px; object-fit:cover; border-radius:6px; border:1px solid var(--border); flex-shrink:0;';

            const info = document.createElement('div');
            info.style.cssText = 'text-align:left; flex:1; min-width:0;';

            const nameEl = document.createElement('div');
            nameEl.style.cssText = 'font-size:12px; font-weight:500; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
            nameEl.textContent = file.name;
            nameEl.title = file.name;

            const sizeEl = document.createElement('div');
            sizeEl.style.cssText = 'font-size:10px; color:var(--text-muted);';
            sizeEl.textContent = `${(file.size / 1024).toFixed(1)} KB`;

            info.appendChild(nameEl);
            info.appendChild(sizeEl);

            const removeOne = document.createElement('button');
            removeOne.type = 'button';
            removeOne.className = 'remove-one-btn';
            removeOne.title = 'ลบรูปนี้';
            removeOne.innerHTML = '<i class="fa-solid fa-xmark"></i>';
            removeOne.style.cssText = [
                'border:none', 'background:transparent', 'color:var(--text-muted)',
                'cursor:pointer', 'font-size:14px', 'padding:4px 6px',
                'border-radius:6px', 'line-height:1',
                'flex-shrink:0', 'margin-left:auto'
            ].join(';');
            removeOne.addEventListener('mouseenter', () => { removeOne.style.color = '#dc2626'; });
            removeOne.addEventListener('mouseleave', () => { removeOne.style.color = 'var(--text-muted)'; });
            removeOne.addEventListener('click', (e) => {
                e.stopPropagation();
                removeFileAt(index);
            });

            imgCard.appendChild(thumb);
            imgCard.appendChild(info);
            imgCard.appendChild(removeOne);
            imagePreviewGrid.appendChild(imgCard);
        });

        const hasFiles = selectedFiles.length > 0;
        previewContainer.classList.toggle('hidden', !hasFiles);
        processBtn.disabled = !hasFiles;
        if (!hasFiles) fileInput.value = '';
    }

    function removeFileAt(index) {
        selectedFiles.splice(index, 1);
        renderPreviews();
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFiles = [];
        isSampleFile = false;
        sampleFileName = '';
        fileInput.value = '';
        renderPreviews();
        previewContainer.classList.add('hidden');
        processBtn.disabled = true;
        chipBtns.forEach(c => c.classList.remove('active'));
    }

    // ── 7. Workspace Sample Chips ─────────────────────────────
    chipBtns.forEach(chip => {
        chip.addEventListener('click', (e) => {
            e.stopPropagation();
            const fileName = chip.dataset.file;

            chipBtns.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            selectedFiles = [];
            isSampleFile = true;
            sampleFileName = fileName;

            imagePreviewGrid.innerHTML = '';
            imagePreviewGrid.style.gridTemplateColumns = '1fr';

            const card = document.createElement('div');
            card.className = 'chip active';
            card.style.cssText = 'padding:6px 8px; cursor:default; display:flex; align-items:center; gap:8px; min-width:0;';

            const thumb = document.createElement('img');
            thumb.src = `/static/samples/${encodeURIComponent(fileName)}`;
            thumb.style.cssText = 'width:40px; height:40px; object-fit:cover; border-radius:6px; border:1px solid var(--border); flex-shrink:0;';

            const label = document.createElement('span');
            label.style.cssText = 'font-size:12px; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
            label.textContent = `${fileName} (Sample)`;

            card.appendChild(thumb);
            card.appendChild(label);
            imagePreviewGrid.appendChild(card);

            previewContainer.classList.remove('hidden');
            processBtn.disabled = false;
        });
    });

    // ── 8. Run OCR Engine ─────────────────────────────────────
    processBtn.addEventListener('click', runOcr);
    retryBtn.addEventListener('click', runOcr);

    async function runOcr() {
        if (selectedFiles.length === 0 && !isSampleFile) return;

        showOnly(resultsLoading);
        processBtn.disabled = true;

        let url, options;
        if (isSampleFile) {
            url = `/api/ocr/sample?filename=${encodeURIComponent(sampleFileName)}`;
            options = { method: 'POST' };
        } else {
            const fd = new FormData();
            selectedFiles.forEach(file => fd.append('files', file));
            url = '/api/ocr';
            options = { method: 'POST', body: fd };
        }

        try {
            const res = await fetch(url, options);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'การวิเคราะห์ล้มเหลว ตรวจสอบสถานะการ์ดจอหลังบ้าน');
            }
            const data = await res.json();
            currentResult = data;
            await renderResults(data);
        } catch (err) {
            console.error(err);
            errorText.textContent = err.message || 'เกิดข้อผิดพลาดที่ไม่คาดคิดในการติดต่อเซิร์ฟเวอร์';
            showOnly(resultsError);
        } finally {
            processBtn.disabled = false;
        }
    }

    // ── 9. Render Results: ช่องแก้ค่าทีละหลัก ─────────────────
    // ตาราง label/หน่วย/ทศนิยมมาจาก server (/api/reading-fields) — แหล่งเดียวกับที่ใช้บันทึก
    // ทุกค่าเป็น "สตริงตัวเลข" ตลอดทาง ไม่แปลงเป็น Number (กันปัดเศษ/เติม .0)
    const fieldEditor = document.getElementById('fieldEditor');
    const recordedAtInput = document.getElementById('recordedAt');
    const prevReadingInfo = document.getElementById('prevReadingInfo');

    let fieldDefs = [];
    let digitBoxes = 10;
    let turbineKey = 'turbine_v_m3';
    let prevReading = null;
    const editor = {};   // key -> { def, initial, negative, boxes[], els{}, state{} }
    let trimRight = null, trimLeft = null;
    let savedReading = null;  // บันทึกแล้ว -> ล็อกช่อง (ไม่มีฟังก์ชันแก้ข้อมูลที่บันทึกไปแล้ว)

    const exportBtn = document.getElementById('exportBtn');
    const exportStatus = document.getElementById('exportStatus');
    const recordedAtWarning = document.getElementById('recordedAtWarning');
    recordedAtInput.addEventListener('input', updateExportState);

    const fieldsReady = fetch('/api/reading-fields')
        .then(res => res.json())
        .then(data => {
            fieldDefs = data.fields;
            digitBoxes = data.digit_boxes;
            turbineKey = data.turbine_field;
        });

    async function loadPrevReading() {
        try {
            const res = await fetch(`/api/meter-runs/${runId}/latest-reading`);
            prevReading = res.ok ? await res.json() : null;
        } catch (err) {
            console.error(err);
            prevReading = null;
        }
        renderPrevInfo();
    }

    function renderPrevInfo() {
        prevReadingInfo.textContent = prevReading
            ? `ค่าจดครั้งก่อน: ${formatDateTime(prevReading.recorded_at)}`
            : 'ยังไม่มีประวัติการจดของ Meter Run นี้';
    }

    function prevValueOf(key) {
        const v = prevReading && prevReading.final_values ? prevReading.final_values[key] : null;
        return v === undefined || v === null || v === '' ? null : String(v);
    }

    async function renderResults(data) {
        document.getElementById('valMeterType').textContent = capitalize(data.meter_type) || '—';
        document.getElementById('valDetectedFormat').textContent = data.detected_format || '—';
        jsonPreCode.textContent = JSON.stringify(data, null, 4);

        await fieldsReady;
        recordedAtInput.value = nowForInput();   // default = ตอนนี้ (แก้ได้) — ตั้งใหม่ทุกครั้งที่อ่านค่า
        recordedAtInput.disabled = false;
        savedReading = null;
        exportStatus.textContent = '';
        exportStatus.className = 'export-status';
        exportBtn.innerHTML = '<i class="fa-solid fa-file-pdf"></i> Export to PDF';
        renderPrevInfo();

        const ed = data.editor || { values: {}, truncated_fields: [], turbine_raw_decimals: 0 };
        const lowConf = new Set(Array.isArray(data.low_confidence_fields) ? data.low_confidence_fields : []);
        const truncated = new Set(ed.truncated_fields || []);
        const raw = data.raw_values || {};
        const rawFallback = data.raw_source === 'fallback';
        document.getElementById('rawSourceWarning').classList.toggle('hidden', !rawFallback);
        document.getElementById('guardWarning').classList.toggle('hidden', !data.guard_applied);

        fieldEditor.innerHTML = '';
        fieldDefs.forEach((def, index) => {
            const row = buildFieldRow(def, {
                initial: ed.values[def.key] ?? null,
                raw: raw[def.key] ?? null,
                lowConf: lowConf.has(def.key),
                truncated: truncated.has(def.key),
                turbineRawDecimals: def.key === turbineKey ? ed.turbine_raw_decimals : 0,
                rawFallback,
                guardDropped: Object.prototype.hasOwnProperty.call(data.guard_dropped || {}, def.key),
            });
            row.style.animationDelay = `${index * 0.04}s`;
            fieldEditor.appendChild(row);
        });
        fieldDefs.forEach(def => updateField(def.key));
        updateExportState();

        showOnly(currentTab === 'table-view' ? resultsTableContainer : resultsJsonContainer);
    }

    function buildFieldRow(def, info) {
        const isTurbine = def.key === turbineKey;
        const d = def.decimals;
        let initial = info.initial;
        const negative = !isTurbine && initial !== null && initial.startsWith('-');
        if (negative) initial = initial.slice(1);

        const row = el('div', 'field-row animate-in');
        row.dataset.key = def.key;

        // หัวแถว: label + หน่วย + ป้าย
        const head = el('div', 'field-head');
        const label = el('div', 'field-label', def.label);
        const unit = el('div', 'field-unit', def.unit);
        const badges = el('div', 'field-badges');
        head.append(label, unit, badges);

        const body = el('div', 'field-body');

        // ค่าจดครั้งก่อน (สีเทา เหนือแถว) — ไม่มีประวัติ = ปล่อยว่าง
        const prev = prevValueOf(def.key);
        const prevEl = el('div', 'prev-value');
        if (prev !== null) {
            prevEl.append(el('span', 'prev-label', 'ค่าจดครั้งก่อน'), el('span', 'prev-number', prev));
        }

        // ช่องทีละหลัก 10 ช่อง + จุดทศนิยมตายตัว
        const digitWrap = el('div', 'digit-wrap');
        const digitRow = el('div', 'digit-row');
        if (negative) digitRow.appendChild(el('span', 'digit-sign', '−'));
        const boxes = [];
        for (let i = 0; i < digitBoxes; i++) {
            if (d > 0 && i === digitBoxes - d) digitRow.appendChild(el('span', 'digit-dot', '.'));
            const box = document.createElement('input');
            box.className = 'digit-box';
            box.inputMode = 'numeric';
            box.autocomplete = 'off';
            box.setAttribute('aria-label', `${def.label} หลักที่ ${i + 1}`);
            attachBoxHandlers(def.key, box, i);
            boxes.push(box);
            digitRow.appendChild(box);
        }
        digitWrap.appendChild(digitRow);

        const els = { row, badges, notes: el('div', 'field-notes') };

        // checkbox ตัดทศนิยม Turbine: อยู่ใต้ 2 ช่องขวาสุดพอดี
        if (isTurbine) {
            const trimRow = el('div', 'trim-row');
            trimLeft = makeTrimCheckbox('ตัด 2 หลัก');
            trimRight = makeTrimCheckbox('ตัด 1 หลัก');
            trimRow.append(trimLeft.wrap, trimRight.wrap);
            digitWrap.appendChild(trimRow);

            // ติ๊กซ้าย (2 หลัก) -> บังคับติ๊กขวาด้วย | เอาขวาออก -> ซ้ายต้องออกด้วย
            trimLeft.input.addEventListener('change', () => {
                if (trimLeft.input.checked) trimRight.input.checked = true;
                updateField(def.key);
            });
            trimRight.input.addEventListener('change', () => {
                if (!trimRight.input.checked) trimLeft.input.checked = false;
                updateField(def.key);
            });

            els.finalPreview = el('div', 'final-preview');
        }

        body.append(prevEl, digitWrap);
        if (els.finalPreview) body.appendChild(els.finalPreview);
        body.appendChild(els.notes);
        row.append(head, body);

        // เติมค่าเริ่มต้นจากโมเดล (ชิดขวา) — ถ้ายาวเกินช่อง ไม่ตัดหลักซ้ายทิ้งเงียบๆ
        const overflow = initial !== null && !fillBoxes(boxes, initial, d);

        editor[def.key] = {
            def, boxes, els, negative, isTurbine,
            initial: overflow ? null : initial,
            raw: info.raw,
            lowConf: info.lowConf,
            truncated: info.truncated,
            overflow,
            turbineRawDecimals: info.turbineRawDecimals,
            rawFallback: info.rawFallback,
            guardDropped: info.guardDropped,
        };
        return row;
    }

    function makeTrimCheckbox(title) {
        const wrap = el('label', 'trim-cell');
        wrap.title = title;
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.setAttribute('aria-label', title);
        wrap.appendChild(input);
        return { wrap, input };
    }

    // เติมสตริง (ไม่มีเครื่องหมายลบ) ลงช่อง ชิดขวา: ส่วนเต็มชิดขวาของช่องหน้าจุด, ทศนิยมหลังจุด
    // คืน false ถ้ายาวเกินจำนวนช่อง (ไม่เติมอะไรเลย)
    function fillBoxes(boxes, value, d) {
        const [intPart, fracPart = ''] = value.split('.');
        const intSlots = boxes.length - d;
        if (intPart.length > intSlots || fracPart.length > d) return false;
        boxes.forEach(b => { b.value = ''; });
        const intDigits = intPart.split('');
        intDigits.forEach((ch, i) => { boxes[intSlots - intDigits.length + i].value = ch; });
        fracPart.split('').forEach((ch, i) => { boxes[intSlots + i].value = ch; });
        return true;
    }

    function attachBoxHandlers(key, box, index) {
        box.addEventListener('focus', () => box.select());

        // สำรองสำหรับคีย์บอร์ดมือถือ (keydown ได้ key = 'Unidentified') — เอาตัวเลขที่เพิ่งพิมพ์
        box.addEventListener('input', (e) => {
            const typed = e.data ? e.data.replace(/\D/g, '') : '';
            const digit = (typed || box.value.replace(/\D/g, '')).slice(-1);
            box.value = digit;
            if (digit) focusBox(key, index + 1);
            updateField(key);
        });

        box.addEventListener('keydown', (e) => {
            // พิมพ์ตัวเลข = แทนที่ค่าในช่องเสมอ ไม่ว่า caret อยู่ตรงไหน (ไม่พึ่ง select() ที่ mouseup ล้างได้)
            if (/^[0-9]$/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                box.value = e.key;
                focusBox(key, index + 1);
                updateField(key);
            } else if (e.key === 'Delete') {
                e.preventDefault();
                box.value = '';
                updateField(key);
            } else if (e.key === 'Backspace' && box.value === '') {
                e.preventDefault();
                const prevBox = editor[key].boxes[index - 1];
                if (prevBox) { prevBox.value = ''; prevBox.focus(); updateField(key); }
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault(); focusBox(key, index - 1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault(); focusBox(key, index + 1);
            }
        });

        // วางตัวเลขทั้งค่า (เช่น 1234.5) ลงช่องไหนก็ได้ของฟิลด์ -> เติมใหม่ทั้งแถว
        box.addEventListener('paste', (e) => {
            e.preventDefault();
            const text = (e.clipboardData || window.clipboardData).getData('text').trim();
            const f = editor[key];
            const value = normalizePasted(text, f.def.decimals, f.isTurbine);
            if (value === null || !fillBoxes(f.boxes, value, f.def.decimals)) {
                flashNote(key, `วางค่า "${text}" ไม่ได้ — ต้องเป็นตัวเลขที่พอดีกับจำนวนช่อง`);
                return;
            }
            updateField(key);
        });
    }

    function focusBox(key, index) {
        const box = editor[key].boxes[index];
        if (box) box.focus();
    }

    function normalizePasted(text, d, isTurbine) {
        const clean = text.replace(/[,\s]/g, '');
        if (!/^\d+(\.\d+)?$/.test(clean)) return null;
        if (isTurbine) return clean.replace('.', '').replace(/^0+(?=\d)/, '');
        let [intPart, fracPart = ''] = clean.split('.');
        intPart = intPart.replace(/^0+(?=\d)/, '');
        fracPart = fracPart.slice(0, d).padEnd(d, '0');
        return d > 0 ? `${intPart}.${fracPart}` : intPart;
    }

    // อ่านค่าจากช่อง -> { value: สตริงหรือ null, error: ข้อความหรือ null }
    function composeValue(f) {
        const d = f.def.decimals;
        const vals = f.boxes.map(b => b.value);
        if (vals.every(v => v === '')) return { value: null, error: null };

        const intVals = vals.slice(0, vals.length - d);
        const fracVals = vals.slice(vals.length - d);
        const first = intVals.findIndex(v => v !== '');
        if (first === -1) return { value: null, error: 'ต้องมีตัวเลขหน้าจุดทศนิยมอย่างน้อย 1 หลัก' };
        if (intVals.slice(first).some(v => v === '')) return { value: null, error: 'มีช่องว่างคั่นกลางตัวเลข' };
        if (fracVals.some(v => v === '')) return { value: null, error: `กรอกทศนิยมให้ครบ ${d} หลัก` };

        const intPart = intVals.slice(first).join('').replace(/^0+(?=\d)/, '');
        let value = d > 0 ? `${intPart}.${fracVals.join('')}` : intPart;
        if (f.negative && /[1-9]/.test(value)) value = `-${value}`;
        return { value, error: null };
    }

    // ตัดหลักขวาสุดตาม checkbox — ทำบนสตริงตัวเลข (เหมือน trim_turbine ฝั่ง server)
    function trimTurbine(digits, n) {
        if (digits === null || !/^\d+$/.test(digits)) return null;
        if (n === 0) return digits.replace(/^0+(?=\d)/, '');
        if (digits.length <= n) return null;
        return digits.slice(0, -n).replace(/^0+(?=\d)/, '');
    }

    function turbineTrimDigits() {
        if (!trimRight || !trimRight.input.checked) return 0;
        return trimLeft.input.checked ? 2 : 1;
    }

    function updateField(key) {
        const f = editor[key];
        if (!f) return;
        const { value, error } = composeValue(f);
        const initialSigned = f.initial === null ? null : (f.negative ? `-${f.initial}` : f.initial);

        let finalValue = value;
        let trim = 0;
        if (f.isTurbine) {
            trim = turbineTrimDigits();
            finalValue = value === null ? null : trimTurbine(value, trim);
            // แสดงช่องที่จะถูกตัดให้เห็น
            f.boxes.forEach((b, i) => b.classList.toggle('cut', i >= f.boxes.length - trim));
        }

        // ถูกแก้ = ค่าในช่องต่างจากค่าเริ่มต้นของโมเดล (เทียบสตริงที่ normalize แล้วทั้งคู่)
        // ช่องที่กรอกไม่ครบ (error) นับว่าแก้แล้วเสมอ
        const edited = error ? true : value !== initialSigned;
        f.state = { value, finalValue, error, edited, trim };

        // ── ป้าย ──
        f.els.badges.innerHTML = '';
        if (f.lowConf) {
            f.els.badges.appendChild(badge('warn', 'fa-triangle-exclamation', 'ตรวจสอบ',
                'โมเดลอ่านได้แต่ไม่มั่นใจ (แสงสะท้อน/ฝุ่น) กรุณาตรวจสอบ'));
        }
        if (f.state.edited) {
            f.els.badges.appendChild(badge('edited', 'fa-pen', 'แก้ไขแล้ว', 'ค่านี้ถูกแก้ด้วยมือ'));
        }
        if (value === null && !error && f.initial === null) {
            // ตาข่าย analog ตัดค่านี้ทิ้ง (ไม่ใช่โมเดลอ่านไม่ออก) -> ป้ายต่างกัน ไม่ให้เข้าใจผิด
            f.els.badges.appendChild(f.guardDropped
                ? badge('null', 'fa-scissors', 'ตัดออก', 'คำขอนี้มีแต่ภาพ analog ค่าของ EVC จึงถูกตัดออก — กรอกเองได้')
                : badge('null', null, 'ไม่พบค่า', 'โมเดลไม่เห็นค่านี้ในภาพ — กรอกเองได้'));
        }

        // ── หมายเหตุ/คำเตือน ──
        const notes = [];
        if (error) notes.push(['error', error]);
        if (f.overflow) {
            notes.push(['error', `ค่าที่โมเดลอ่าน (${f.raw}) ยาวเกินจำนวนช่อง — กรุณากรอกเอง`]);
        }
        // ตัดทศนิยมเกินเป็นพฤติกรรมปกติตามสเปค -> ไม่เตือนบนจอ (log ฝั่ง server แทน)
        // เตือนทุกครั้งทำให้ผู้ใช้เลิกอ่านคำเตือนที่สำคัญจริง
        if (f.isTurbine && f.turbineRawDecimals > 0) {
            notes.push(['info', `โมเดลอ่านได้ ${f.raw} (ทศนิยม ${f.turbineRawDecimals} หลัก) — ตรวจว่าติ๊กตัดทศนิยมถูกต้อง`]);
        }
        if (f.isTurbine && f.rawFallback && f.initial !== null) {
            notes.push(['warn', 'ค่านี้มาจากทางสำรอง — ถ้าช่องแดงหลักท้ายเป็นเลข 0 อาจหายไป เทียบจำนวนหลักกับภาพก่อนติ๊กตัด']);
        }
        if (f.isTurbine && value !== null && finalValue === null) {
            notes.push(['error', 'ตัดหลักจนไม่เหลือตัวเลข']);
        }
        if (f.negative) {
            notes.push(['info', 'ค่าติดลบ (ตามที่โมเดลอ่าน) — แก้ได้เฉพาะตัวเลข']);
        }
        if (f.flash) notes.push(['error', f.flash]);
        f.flash = null;

        f.els.notes.innerHTML = '';
        notes.forEach(([kind, text]) => f.els.notes.appendChild(el('div', `field-note ${kind}`, text)));

        f.els.row.classList.toggle('has-error', !!error || (f.isTurbine && value !== null && finalValue === null));

        if (f.els.finalPreview) {
            f.els.finalPreview.innerHTML = '';
            f.els.finalPreview.append(
                el('span', 'final-label', 'ค่าที่จะบันทึก'),
                el('span', 'final-number', finalValue ?? '—'),
                el('span', 'final-unit', f.def.unit),
            );
        }
        updateExportState();
        scheduleDecreaseCheck();
    }

    // ── 9.4 เตือนผลต่างค่าสะสมติดลบ ──────────────────────────
    // server ตัดสิน (compute_final_values + decreased_fields ตัวเดียวกับตอนบันทึก/PDF)
    // เตือนอย่างเดียว ไม่บล็อกปุ่ม Export
    const decreaseWarningEl = document.getElementById('decreaseWarning');
    const decreaseList = document.getElementById('decreaseList');
    let decreaseTimer = null;
    let decreaseSeq = 0;

    function scheduleDecreaseCheck() {
        clearTimeout(decreaseTimer);
        decreaseTimer = setTimeout(checkDecrease, 200);
    }

    async function checkDecrease() {
        const seq = ++decreaseSeq;
        if (!prevReading || !currentRun || savedReading) {
            if (!savedReading) renderDecrease([]);
            return;
        }
        try {
            const res = await fetch(`/api/meter-runs/${currentRun.id}/decrease-check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    entered_values: collectEntered(),
                    turbine_trim_digits: turbineTrimDigits(),
                }),
            });
            if (seq !== decreaseSeq) return;   // มีการแก้ค่าใหม่ระหว่างรอ -> ทิ้งผลเก่า
            // 400 = ค่าในช่องยังไม่ครบ/ผิดรูปแบบ (ช่องแดงเตือนอยู่แล้ว) -> ไม่แสดง
            renderDecrease(res.ok ? (await res.json()).decreased : []);
        } catch (err) {
            console.error(err);
        }
    }

    function renderDecrease(items) {
        decreaseList.innerHTML = '';
        items.forEach(item => {
            const li = el('li', null, item.message);
            li.appendChild(el('span', 'decrease-detail',
                `ครั้งนี้ ${item.current} · ครั้งก่อน ${item.previous} · ผลต่าง ${item.difference}`));
            decreaseList.appendChild(li);
        });
        decreaseWarningEl.classList.toggle('hidden', items.length === 0);
    }

    // ── 9.5 Export to PDF: บันทึกลง DB + ดาวน์โหลด PDF ในครั้งเดียว ──────
    // server คำนวณ final_values เองจากค่าในช่อง (ไม่ส่งค่าที่ตัดแล้วไป) -> ค่าที่บันทึกกับ PDF ตรงกันเสมอ

    // เหตุผลที่ยัง export ไม่ได้ (null = พร้อม)
    function exportBlocker() {
        if (!currentRun) return 'ยังไม่ได้เลือก Meter Run';
        if (!currentResult) return 'ยังไม่มีผลการอ่านค่า';
        if (!currentResult.request_id) return 'เก็บผลการอ่านไม่สำเร็จ — กรุณาอ่านค่าใหม่';
        if (!recordedAtInput.value) return 'กรุณากรอกวัน/เวลาที่จด';
        const states = Object.values(editor).map(f => f.state).filter(Boolean);
        if (states.some(st => st.error || (st.value !== null && st.finalValue === null))) {
            return 'แก้ช่องที่เป็นสีแดงก่อน';
        }
        if (states.every(st => st.finalValue === null)) return 'ไม่มีค่าให้บันทึก';
        return null;
    }

    function updateExportState() {
        // วัน/เวลาที่จดเก่ากว่าครั้งก่อน -> PDF จะใช้ "ครั้งก่อนหน้าวันนั้น" ไม่ใช่ค่าที่แสดงอยู่
        const older = prevReading && recordedAtInput.value &&
            recordedAtInput.value < prevReading.recorded_at.slice(0, 16);
        recordedAtWarning.textContent = older
            ? 'วัน/เวลาที่จดเก่ากว่าการจดครั้งก่อน — ค่าครั้งก่อนใน PDF จะเป็นของการจดก่อนหน้าวันนั้น ไม่ใช่ค่าที่แสดงอยู่'
            : '';
        recordedAtWarning.classList.toggle('hidden', !older);

        if (savedReading) return;  // บันทึกแล้ว ปุ่มอยู่ในสถานะ "บันทึกแล้ว"
        const blocker = exportBlocker();
        exportBtn.disabled = blocker !== null;
        exportBtn.title = blocker || '';
    }

    function collectEntered() {
        const entered = {};
        Object.entries(editor).forEach(([key, f]) => { entered[key] = f.state ? f.state.value : null; });
        return entered;
    }

    exportBtn.addEventListener('click', async () => {
        if (exportBlocker() !== null || savedReading) return;
        exportBtn.disabled = true;
        setExportStatus('info', 'กำลังบันทึก…');
        try {
            const res = await fetch('/api/readings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    meter_run_id: currentRun.id,
                    request_id: currentResult.request_id,
                    recorded_at: recordedAtInput.value,
                    entered_values: collectEntered(),
                    turbine_trim_digits: turbineTrimDigits(),
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(typeof data.detail === 'string' ? data.detail : 'บันทึกไม่สำเร็จ');
            }
            savedReading = data.reading;
            lockEditor();
            downloadPdf(data.pdf_url);
            showSaved(data.reading, data.pdf_url);
        } catch (err) {
            console.error(err);
            setExportStatus('error', err.message || 'เกิดข้อผิดพลาดในการติดต่อเซิร์ฟเวอร์');
            updateExportState();
        }
    });

    function setExportStatus(kind, text) {
        exportStatus.className = `export-status ${kind}`;
        exportStatus.textContent = text;
    }

    function showSaved(reading, pdfUrl) {
        exportStatus.className = 'export-status success';
        exportStatus.textContent = '';
        const icon = document.createElement('i');
        icon.className = 'fa-solid fa-circle-check';
        const link = document.createElement('a');
        link.href = pdfUrl;
        link.textContent = 'ดาวน์โหลด PDF อีกครั้ง';
        exportStatus.append(icon, ` บันทึกแล้ว · รายการที่ #${reading.id} · `, link);
        exportBtn.disabled = true;
        exportBtn.innerHTML = '<i class="fa-solid fa-check"></i> บันทึกแล้ว';
    }

    function downloadPdf(url) {
        const a = document.createElement('a');
        a.href = url;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    // บันทึกแล้วห้ามแก้ (นอกขอบเขต: ไม่ทำฟังก์ชันแก้ข้อมูลที่บันทึกไปแล้ว) — อ่านค่าใหม่เพื่อจดครั้งถัดไป
    function lockEditor() {
        Object.values(editor).forEach(f => f.boxes.forEach(b => { b.disabled = true; }));
        if (trimLeft) trimLeft.input.disabled = true;
        if (trimRight) trimRight.input.disabled = true;
        recordedAtInput.disabled = true;
    }

    function flashNote(key, msg) {
        editor[key].flash = msg;
        updateField(key);
    }

    function badge(kind, icon, text, title) {
        const b = el('span', `status-badge ${kind}`);
        if (title) b.title = title;
        if (icon) {
            const i = document.createElement('i');
            i.className = `fa-solid ${icon}`;
            b.appendChild(i);
            b.appendChild(document.createTextNode(' '));
        }
        b.appendChild(document.createTextNode(text));
        return b;
    }

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function nowForInput() {
        const d = new Date();
        d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
        return d.toISOString().slice(0, 16);
    }

    function formatDateTime(iso) {
        if (!iso) return '—';
        const [date, time = ''] = iso.split('T');
        const [y, m, d] = date.split('-');
        return `${d}/${m}/${y} ${time.slice(0, 5)}`;
    }

    // ── 10. Copy JSON ─────────────────────────────────────────
    copyJsonBtn.addEventListener('click', async () => {
        if (!currentResult) return;
        try {
            await navigator.clipboard.writeText(JSON.stringify(currentResult, null, 4));
            const orig = copyJsonBtn.innerHTML;
            copyJsonBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            setTimeout(() => { copyJsonBtn.innerHTML = orig; }, 2000);
        } catch (e) {
            console.error('Copy failed', e);
        }
    });

    // ── 11. Utilities ─────────────────────────────────────────
    function capitalize(s) {
        if (!s) return '';
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

});