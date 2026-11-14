// Generates the MedicalVQA IEEE paper by re-using the ManuMind paper's package (styles, page setup, footer).
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const TEMPLATE_DIR = process.argv[2];           // unzipped ManuMind docx
const OUT_DIR = path.join(__dirname, '_pkg');
const OUT_DOCX = process.argv[3];

const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// *italic* markup -> runs
function runs(text, base = '') {
  const parts = text.split(/(\*[^*]+\*)/g).filter(Boolean);
  return parts.map(p => {
    const it = p.startsWith('*') && p.endsWith('*');
    const t = it ? p.slice(1, -1) : p;
    const rpr = (it ? '<w:i/><w:iCs/>' : '') + base;
    return `<w:r>${rpr ? `<w:rPr>${rpr}</w:rPr>` : ''}<w:t xml:space="preserve">${esc(t)}</w:t></w:r>`;
  }).join('');
}
const P = (style, text, extra = '') =>
  `<w:p><w:pPr><w:pStyle w:val="${style}"/>${extra}</w:pPr>${runs(text)}</w:p>`;
const H1 = t => P('Heading1', t);
const H2 = t => P('Heading2', t);
const B = t => P('BodyText', t);
const H5 = t => P('Heading5', t);
const REF = t => P('references', t);

let eqNo = 0;
const EQ = t => {
  eqNo++;
  const f = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>';
  return `<w:p><w:pPr><w:pStyle w:val="equation"/></w:pPr><w:r><w:tab/></w:r><w:r><w:rPr>${f}<w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">${esc(t)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:rPr>${f}<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">(${eqNo})</w:t></w:r></w:p>`;
};

function TABLE(title, widths, header, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, style, hdr) =>
    `<w:tc><w:tcPr><w:tcW w:w="${w}" w:type="dxa"/>${hdr ? '<w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>' : ''}</w:tcPr><w:p><w:pPr><w:pStyle w:val="${style}"/><w:spacing w:before="1pt" w:after="1pt"/><w:jc w:val="start"/></w:pPr>${runs(txt)}</w:p></w:tc>`;
  const tr = (cells, style, hdr) =>
    `<w:tr><w:trPr><w:cantSplit/></w:trPr>${cells.map((c, i) => cell(c, widths[i], style, hdr)).join('')}</w:tr>`;
  return `<w:p><w:pPr><w:pStyle w:val="tablehead"/><w:keepNext/></w:pPr>${runs(title)}</w:p>` +
    `<w:tbl><w:tblPr><w:tblW w:w="${total}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:start w:w="3pt" w:type="dxa"/><w:end w:w="3pt" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>${widths.map(w => `<w:gridCol w:w="${w}"/>`).join('')}</w:tblGrid>` +
    tr(header, 'tablecolhead', true) + rows.map(r => tr(r, 'tablecopy', false)).join('') + `</w:tbl>` +
    `<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p>`;
}

// figures: picture paragraph + auto-numbered caption (figurecaption style carries the "Fig. N." numbering)
const FIGS = [];
function FIG(file, pxW, pxH, caption) {
  const n = FIGS.length + 1;
  FIGS.push({ file, rid: `rIdFig${n}`, media: `fig${n}.png` });
  const cx = 3150000, cy = Math.round(cx * pxH / pxW), id = 200 + n;
  return `<w:p><w:pPr><w:keepNext/><w:spacing w:before="6pt" w:after="0pt"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="${id}" name="Figure ${n}" descr="${esc(caption)}"/><a:graphic xmlns:a="http://purl.oclc.org/ooxml/drawingml/main"><a:graphicData uri="http://purl.oclc.org/ooxml/drawingml/picture"><pic:pic xmlns:pic="http://purl.oclc.org/ooxml/drawingml/picture"><pic:nvPicPr><pic:cNvPr id="${id}" name="fig${n}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdFig${n}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>` +
    `<w:p><w:pPr><w:pStyle w:val="figurecaption"/><w:jc w:val="center"/></w:pPr>${runs(caption)}</w:p>`;
}

// ------------------------------------------------------------------ CONTENT
const title = 'Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images: Pipeline Implementation and Pilot Results with General and Medical Vision-Language Models';

const authorRun = (n, e) =>
  `<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">${esc(n)}</w:t></w:r>` +
  (e ? `<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve"> (${esc(e)})</w:t></w:r>` : '');
const sep = `<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">, </w:t></w:r>`;

const head =
  `<w:p><w:pPr><w:pStyle w:val="papertitle"/></w:pPr><w:r><w:t xml:space="preserve">${esc(title)}</w:t></w:r></w:p>` +
  `<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="14pt" w:after="4pt"/></w:pPr>` +
  [authorRun('Abisha K M', 'abishakm1507@gmail.com'), authorRun('Anusiya S', 'anusiya2705@gmail.com'),
   authorRun('Bhavya Sree P R', 'bhavyasree.050305@gmail.com'), authorRun('Dirisala Kavya', '')].join(sep) + `</w:p>` +
  `<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="6pt" w:after="10pt"/></w:pPr><w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">Department of Artificial Intelligence and Data Science</w:t></w:r><w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:br/></w:r><w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">R.M.K. Engineering College, Kavaraipettai, Tamil Nadu, India</w:t></w:r></w:p>`;

const abstractText = 'Vision-language models can answer questions about medical images fluently, but they may state findings that are not in the image, give no indication of where a finding is, and rarely express doubt. We have built the software for a pipeline that tries to answer only what it can point to. A detector and MedSAM propose numbered candidate regions, the vision-language model must answer in a JSON contract that cites those regions, a verifier removes claims that cite no valid region, and a confidence score that combines answer agreement, semantic entropy and the localizer score decides whether to answer or abstain. Qwen2.5-VL-3B and MedGemma-4B are the two models behind a common interface. The pipeline and its metric code are implemented and covered by unit tests. The experimental evidence is small. On a 10-question pilot from the SLAKE test split, Qwen2.5-VL-3B answered 6 correctly and MedGemma-4B answered 7, but the two runs differed in more than the model, so the difference cannot be attributed to grounding or abstention. The confidence weights and threshold have not been fitted on real model outputs, the one calibration run used synthetic data, and the integrated pipeline has not been evaluated on the held-out test cohort. The repository’s grounding and calibration metrics for the pilot use placeholder inputs, so we do not report them. No clinical validation was done.';

const body = [];
const push = (...x) => body.push(...x);

push(
  P('Abstract', '*Abstract*—' + abstractText),
  P('Keywords', '*Keywords*—medical visual question answering, vision-language models, visual grounding, uncertainty estimation, selective prediction, MedSAM'),

  H1('Introduction'),
  B('A radiologist who reads an answer from a vision-language model (VLM) has two questions. Is it right, and where in the image does it come from? A plain-text answer addresses neither. The model may name an abnormality that is not on the scan, it gives the reader no region to check, and it sounds just as sure when it is guessing.'),
  B('We want a system that follows one rule: it only claims what it can point to. Every finding must cite a numbered region of the image, a finding that cites no valid region is removed, and when the evidence is weak the system says “uncertain” and does not answer.'),
  B('This paper reports the implementation of that pipeline and the first, small experiments on it. We are explicit about the status of each part. The software is built and tested. The experimental evidence is a 10-question pilot and a cohort audit. The calibration of the confidence gate on real model outputs and the evaluation of the full pipeline on the held-out test cohort have not been done, and we say so wherever a result could be mistaken for them.'),
  B('The contributions are as follows. (i) A six-stage pipeline with a typed JSON contract between the VLM and the verifier, implemented behind one client interface for Qwen2.5-VL-3B and MedGemma-4B. (ii) A runtime verifier that never uses ground truth, so that evaluation cannot leak into prediction. (iii) A confidence gate whose weights and threshold are parameters to be fitted, not constants. (iv) A cohort and split design that is image-disjoint, with the pilot and calibration images excluded from the test cohort. (v) A pilot comparison of the two models, together with a list of what it cannot show.'),
  B('Related work is in Section II. The system and data are in Section III. Section IV gives results and their limits. Sections V and VI give the conclusion and future work.'),

  H1('Related Work'),
  H2('Medical VQA and Datasets'),
  B('SLAKE [1] provides English and Chinese questions on 642 images of several modalities, with pixel masks and bounding boxes for organs and lesions. Because it has region annotations, it is the main benchmark for checking whether a cited region is correct. VQA-RAD [2] contains clinician-written questions on radiology images but has no region annotations, so we use it only as an ungrounded baseline. Chest-radiograph detection sets such as RSNA Pneumonia [3] and VinDr-CXR [4] provide boxes for training and checking a detector.'),
  H2('Grounding and Visual Prompting'),
  B('The Segment Anything Model [5] segments an object from a box or point prompt, and MedSAM [6] adapts it to medical images. We use MedSAM to turn a box into a mask. Set-of-Mark prompting [7] overlays numbered marks on an image so that a language model can refer to regions by number. We use the same device to make the cited regions explicit and checkable.'),
  H2('Uncertainty and Selective Prediction'),
  B('Kuhn et al. [8] estimate uncertainty in language models by sampling several answers, grouping those with the same meaning and computing the entropy over the groups. We use this semantic entropy together with agreement among samples. Calibration is measured by the expected calibration error [9], and abstention is judged by how accuracy changes as coverage is reduced [10]. Hallucination in generative models is surveyed in [11].'),
  B('It is important to say what the confidence score is. It measures how consistent the model is with itself and how strong the localizer evidence is. It is not a probability of being correct until it has been calibrated, and we have not done that on real outputs.'),

  H1('System Description'),
  H2('Overview'),
  B('The pipeline has six stages. (1) The image and question are read. (2) A detector proposes up to k candidate boxes with scores, and MedSAM converts each box into a mask. (3) The regions are drawn on a copy of the image as R1 to Rk. (4) The VLM answers in a JSON contract with an answer, a list of findings, and for each finding a region identifier, a status (present, absent or uncertain) and a confidence. N answers are sampled. (5) The verifier checks each finding against the region set and removes those without a valid region. (6) The gate combines three signals and either returns the majority answer or abstains. The pipeline is drawn in Fig. 1, where dashed boxes are work that has not been done, and the status of each component is in Table I.'),
  FIG('fig1_architecture.png', 1020, 1250, 'The six-stage pipeline. Solid boxes are implemented. Dashed boxes are not yet done.'),
  TABLE('Implementation Status of System Components', [2750, 2050], ['Component', 'Status'], [
    ['Dataset loaders and leakage checks', 'Built; SLAKE checked on real data'],
    ['Faster R-CNN candidate detector', 'Built; no trained CXR weights evaluated'],
    ['MedSAM adapter', 'Built; verified on one SLAKE image'],
    ['Qwen2.5-VL-3B and MedGemma-4B clients', 'Built; run on 10-question pilot'],
    ['Claim extraction, association, verification, filtering', 'Built and unit-tested'],
    ['Semantic clustering and confidence gate', 'Built and unit-tested'],
    ['Metrics, bootstrap, resumable runner', 'Built and unit-tested'],
    ['Gate calibration on real outputs', 'Not done'],
    ['Integrated pipeline on held-out test cohort', 'Not done'],
    ['Web interface, expert review', 'Specified, not evaluated'],
  ]),
  H2('Datasets and Cohorts'),
  B('All experiments so far use SLAKE. The English test split has 1,061 questions and the English validation split has 1,053. Two kinds of pilot material are kept out of every later evaluation. The first is the 10-question pilot described in Section IV-B, which lies on 9 test images. Removing these 10 questions and every other question on those 9 images leaves 945 eligible test questions on 87 images, of which 572 are open and 373 are closed. A total of 864 of these questions have non-empty region annotations and 81 have none. The second is a 15-image feasibility pilot on the validation split, which has been implemented but not run with real models.'),
  B('For calibration, the 162 validation questions on those 15 images are removed, which leaves 891 questions. They are divided 80/20 by image with a fixed seed, giving 719 questions on 65 images for fitting and 172 questions on 16 images for assessment. No image appears in both parts, and the test split is not used. Splitting by image and not by question matters because several questions share one image, and a question-level split would put the same scan on both sides. The cohorts are shown in Fig. 2. We re-ran the cohort audit on the local copy of SLAKE and obtained the same counts: 1,061 test questions, 945 eligible, 864 with and 81 without region annotations, and 1,053 validation questions.'),
  FIG('fig3_cohorts.png', 1020, 1040, 'Question counts for the test cohort and the calibration split. The test split is never used for calibration.'),
  H2('Localization and Candidate Regions'),
  B('The specification asks for a YOLO- or DETR-style detector. Ultralytics YOLOv8 is released under the AGPL-3.0 licence, which would conflict with the MIT licence of this project, so we chose a Torchvision Faster R-CNN [12] and left room for RT-DETR. MedSAM ViT-B weights were checked by size and MD5 against the reference values, and the adapter produced a valid mask from a box on one SLAKE image.'),
  B('One rule is fixed. If the detector finds nothing above its threshold, the region set is empty. The system never invents boxes, such as image quadrants, to fill the gap. Any positive finding on an empty region set is therefore ungrounded. An empty detector output is also not treated as proof that a finding is absent.'),
  H2('Structured Contract and Claim Verification'),
  B('The VLM must return JSON. Each finding has a status. A “present” finding must cite a region in R. An “absent” finding cites NONE and is handled separately, as are “uncertain” findings, so that negative claims do not distort the grounding metrics. The verifier works on structure only: it checks that a cited identifier exists, and it never reads ground-truth boxes. Ground truth enters only in the evaluation layer. We describe three groups of claims: supported, unsupported, and unverifiable. The unsupported-claim rate (USR) is the fraction of positive claims that are unsupported, and it is reported before filtering (USR_raw) and after (USR_filtered).'),
  H2('Confidence Gate'),
  B('From N = 5 samples at temperature 0.7 the answers are grouped into semantic clusters. Let a be the share of samples in the largest cluster and p_k the share in cluster k. The normalized entropy is'),
  EQ('H_n = −Σ p_k ln p_k / ln N'),
  B('and the fused confidence is'),
  EQ('c = w1·a + w2·(1 − H_n) + w3·ρ'),
  B('with the weights non-negative and summing to one, and ρ the localizer score of the cited region (the minimum over cited regions). For an absent or uncertain answer there is no region, so no localizer score exists and the other two weights are rescaled instead of assigning an artificial value. The decision rule is'),
  EQ('ŷ = y_majority if c ≥ τ, else “uncertain”'),
  B('The weights and τ are chosen by grid search on the calibration-fit images. By design, none of them is hard-coded as a validated value. The calibration script exists, but the only run so far used synthetic validation records, and its output (a weight of 1.0 on ρ and τ = 0.5) is a demonstration of the code path and not a result. It is not used anywhere else in this paper.'),
  H2('Tracks and Conditions'),
  B('Three tracks separate what is being tested. Track A uses detector-proposed regions on chest radiographs. Track B uses the dataset’s ground-truth boxes as oracle regions, which tests the VLM and verifier without detector errors. Track C is ungrounded answering. The project’s SLAKE annotation audit found Track A ineligible on SLAKE, so it is not used here. Three conditions are crossed with the tracks. Condition 1 is a free-form answer with N = 1 and greedy decoding. Condition 2 is a grounded answer with verification and no abstention. Condition 3 adds abstention. Conditions 2 and 3 reuse the same generated samples, since a hash over the model, track, question, image, N, temperature and seed identifies a generation and only the abstention decision differs.'),
  H2('Implementation'),
  B('Qwen2.5-VL-3B is served through Ollama for local runs. MedGemma-4B (google/medgemma-4b-it, a gated model) is loaded through Hugging Face Transformers. For the planned Colab or Kaggle T4 runs, each model is loaded once in 4-bit precision, run on both tracks and then unloaded, so the two models are never in GPU memory together. Each generation is written as an artifact, re-read and validated after writing, and skipped on a restart if it is complete. The development machine has a Core i5-1235U and 16 GB of RAM and no CUDA, so only mocks and tests run there.'),

  H1('Results and Discussion'),
  H2('Evaluation Protocol'),
  B('Answers are compared with the reference after the same normalization on both sides: whitespace, code fences, quotes, case and trailing punctuation are removed, and then an exact match is required. Failed or missing predictions count as wrong and stay in the denominator. This is strict. “Lungs” does not match “Lung”, and one of the open-question errors below is of that kind. All figures were recomputed from the raw prediction files and not copied from earlier summaries.'),
  H2('Pilot Baseline Comparison'),
  B('Ten test questions (five open, five closed) on nine images were answered by both models. The results are in Table II and the paired outcomes in Table III. MedGemma-4B had one more correct answer than Qwen2.5-VL-3B, on closed question 12845 (reference “No”, Qwen answered “yes”). Both models were wrong on three open questions: a count (reference 0), a location (reference “Lower Left Lung”) and the “Lung” versus “lungs” case. One question of difference in ten is not evidence that either model is better. Neither pilot run used regions: both were single answers, MedGemma-4B in bfloat16 on a Kaggle T4 with an answer-only JSON reply, and Qwen2.5-VL-3B through Ollama, and neither was passed through the verifier or the confidence gate. The accuracies are plotted in Fig. 3. We re-ran the comparison script on the raw prediction files and it reproduced 6 of 10 and 7 of 10.'),
  FIG('fig2_pilot_accuracy.png', 1020, 660, 'Pilot accuracy on 10 SLAKE test questions (5 open, 5 closed). The labels give correct answers over questions.'),
  TABLE('Pilot Results on 10 SLAKE Test Questions', [2200, 1300, 1300], ['Metric', 'Qwen2.5-VL-3B', 'MedGemma-4B'], [
    ['Overall accuracy', '60.0% (6/10)', '70.0% (7/10)'],
    ['Open accuracy', '40.0% (2/5)', '40.0% (2/5)'],
    ['Closed accuracy', '80.0% (4/5)', '100.0% (5/5)'],
    ['Failed predictions', '0', '0'],
    ['Mean time per question', '15.9 s', '5.3 s'],
  ]),
  TABLE('Paired Outcomes on the Same 10 Questions', [3000, 1800], ['Outcome', 'Questions'], [
    ['Both correct', '6'],
    ['Both incorrect', '3'],
    ['Only MedGemma-4B correct', '1'],
    ['Only Qwen2.5-VL-3B correct', '0'],
  ]),
  B('The times should not be compared as a speed result either, since the two models were run through different clients and the run conditions were not controlled.'),
  H2('Metrics We Do Not Report'),
  B('The repository’s Phase 5 evaluation script also produces unsupported-claim rate, pointing accuracy, sampling agreement, entropy, ECE, AUROC and an accuracy–coverage curve for the same 10 questions. We do not report them, because on reading the script we found that their inputs are placeholders and not pipeline outputs. The confidence of each answer is set to 0.8 if the answer was correct and 0.3 if it was wrong, so any measure of how well confidence predicts correctness is circular, and an AUROC of 1.0 is guaranteed. Agreement and entropy are fixed constants (0.8 and 0.2) for MedGemma-4B. Pointing accuracy is set equal to correctness, and the claim counts are derived from correctness as well. The existing summary files therefore show that the metric code runs, but they say nothing about grounding, calibration or abstention, and earlier project reports that quote them as exploratory results should be read with this in mind. The script has since been corrected.'),
  H2('Corrections Made During the Work'),
  B('Several errors were found by audit, and we list them because they could easily have gone unnoticed. The cohort script looked for annotation files at the wrong path, so earlier reports counted 1,051 eligible questions. The correct count is 945 once the 9 pilot images are also excluded. A calibration result obtained on synthetic data had been written up as if it came from SLAKE. A model comparison had been described as showing a benefit of grounding. A test count of 227 was corrected to 197. The placeholder confidence described above was found while preparing this paper. It has since been removed: the evaluation script now reports answer accuracy only and marks the other metrics as not computable. An absolute path to another machine in the cohort audit script was also replaced.'),
  H2('Software Verification'),
  B('We ran the test suite in a clean Python 3.11 environment on the development machine. It reported 434 tests with no failures, 2 skipped and 1 collection error. The skipped tests need the MedSAM checkpoint, which is not in the cloned repository, and the error is the detector test module, which needs PyTorch that we did not install. A dry run of the generation harness with the mock client completed on its two fixture questions. That run uses no model, so it shows only that the harness executes. We could not run the real models on this machine: it has no Ollama server, a GPU with 4 GB of memory, and no access token for the gated MedGemma weights. Passing tests show that the code does what it was written to do. They do not show that the pipeline is accurate.'),
  H2('Limitations'),
  B('The evidence is thin, and the main limitations are these. The only model comparison has 10 questions, which is too few for a confidence interval that would mean anything, and the planned bootstrap and McNemar analysis needs the full cohort. The gate has not been fitted on real outputs, so its weights, its threshold and its calibration are unknown, and we have no valid measurement of grounding, calibration or abstention for any model. The integrated pipeline has not been run on the 945-question cohort. Track A, which would test the detector on real chest radiographs, is not available on SLAKE, and no trained detector has been evaluated. SLAKE is a curated benchmark, so performance on routine clinical images should be expected to be worse. Exact-match scoring penalizes correct answers that differ in wording. MedSAM was checked on a single image. Perturbation robustness, masked-region ablation and blinded expert review are specified but not carried out. Nothing here is clinical validation, and the system is not suitable for clinical use.'),

  H1('Conclusion'),
  B('We have implemented a pipeline in which a vision-language model must cite numbered image regions, unsupported claims are removed, and a fused confidence score decides whether to answer or abstain, with Qwen2.5-VL-3B and MedGemma-4B behind one interface. The verifier, the confidence gate and the metrics are built and tested, and the cohorts are image-disjoint. The experimental evidence is still a 10-question pilot, in which MedGemma-4B answered one more question correctly than Qwen2.5-VL-3B under different settings. Whether region grounding and abstention improve the reliability of medical VQA is not yet answered.'),

  H1('Future Work'),
  B('The next steps are in order. First, run the four model and track combinations on the 891-question calibration cohort on a T4 GPU, fit the weights and τ on the fit images, and report ECE, AUROC and the accuracy–coverage curve on the assessment images. Second, run the 945-question test cohort once with the frozen configuration, with bootstrap confidence intervals and McNemar’s test for the paired comparison. Third, train and evaluate a detector on RSNA Pneumonia and VinDr-CXR so that Track A can be tested. Fourth, run the perturbation and masked-region ablation suites. Fifth, add VQA-RAD as an ungrounded baseline and run the blinded review. Last, finish and test the web interface.'),

  H5('Acknowledgment'),
  B('The authors thank the Department of Artificial Intelligence and Data Science at R.M.K. Engineering College and our mentor, Mr. Sathesh Abraham Leo E, Assistant Professor, for their support with this project.'),
  H5('References'),
  REF('B. Liu, L.-M. Zhan, L. Xu, L. Ma, Y. Yang, and X.-M. Wu, “SLAKE: A semantically-labeled knowledge-enhanced dataset for medical visual question answering,” in *Proc. IEEE Int. Symp. Biomedical Imaging*, 2021, pp. 1650–1654.'),
  REF('J. J. Lau, S. Gayen, A. Ben Abacha, and D. Demner-Fushman, “A dataset of clinically generated visual questions and answers about radiology images,” *Scientific Data*, vol. 5, art. 180251, 2018.'),
  REF('G. Shih et al., “Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia,” *Radiology: Artificial Intelligence*, vol. 1, no. 1, art. e180041, 2019.'),
  REF('H. Q. Nguyen et al., “VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations,” *Scientific Data*, vol. 9, art. 429, 2022.'),
  REF('A. Kirillov et al., “Segment anything,” in *Proc. IEEE/CVF Int. Conf. Computer Vision*, 2023, pp. 4015–4026.'),
  REF('J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” *Nature Communications*, vol. 15, art. 654, 2024.'),
  REF('J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-of-Mark prompting unleashes extraordinary visual grounding in GPT-4V,” arXiv:2310.11441, 2023.'),
  REF('L. Kuhn, Y. Gal, and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” in *Proc. Int. Conf. Learning Representations*, 2023.'),
  REF('C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. Int. Conf. Machine Learning*, 2017, pp. 1321–1330.'),
  REF('Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4878–4887.'),
  REF('Z. Ji et al., “Survey of hallucination in natural language generation,” *ACM Computing Surveys*, vol. 55, no. 12, art. 248, 2023.'),
  REF('S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in *Advances in Neural Information Processing Systems*, vol. 28, 2015, pp. 91–99.'),
);

// ------------------------------------------------------------------ PACKAGE
fs.rmSync(OUT_DIR, { recursive: true, force: true });
fs.cpSync(TEMPLATE_DIR, OUT_DIR, { recursive: true });

const tpl = fs.readFileSync(path.join(TEMPLATE_DIR, 'word/document.xml'), 'utf8');
const bodyStart = tpl.indexOf('<w:body>') + '<w:body>'.length;
const prefix = tpl.slice(0, bodyStart);
// first-section break paragraph (single-column title block) and final sectPr, both taken from the template
const m = tpl.match(/<w:p><w:pPr><w:sectPr[\s\S]*?<\/w:sectPr><\/w:pPr><\/w:p>/);
const finalSect = tpl.match(/<w:sectPr><w:type w:val="continuous"\/>[\s\S]*?<\/w:sectPr><\/w:body><\/w:document>$/)[0];
const doc = prefix + head + m[0] + body.join('') + finalSect;
fs.writeFileSync(path.join(OUT_DIR, 'word/document.xml'), doc);

// drop the unused architecture image and its relationship
try {
  const relsPath = path.join(OUT_DIR, 'word/_rels/document.xml.rels');
  let rels = fs.readFileSync(relsPath, 'utf8');
  rels = rels.replace(/<Relationship [^>]*relationships\/image"[^>]*\/>/g, '');
  fs.rmSync(path.join(OUT_DIR, 'word/media'), { recursive: true, force: true });
  fs.mkdirSync(path.join(OUT_DIR, 'word/media'), { recursive: true });
  const add = FIGS.map(f => {
    fs.copyFileSync(path.join(__dirname, 'figs', f.file), path.join(OUT_DIR, 'word/media', f.media));
    return `<Relationship Id="${f.rid}" Type="http://purl.oclc.org/ooxml/officeDocument/relationships/image" Target="media/${f.media}"/>`;
  }).join('');
  rels = rels.replace('</Relationships>', add + '</Relationships>');
  fs.writeFileSync(relsPath, rels);
} catch (e) { console.error('rels cleanup', e.message); }

// title metadata
try {
  const cp = path.join(OUT_DIR, 'docProps/core.xml');
  let c = fs.readFileSync(cp, 'utf8');
  c = c.replace(/<dc:title>[\s\S]*?<\/dc:title>/, `<dc:title>${esc(title)}</dc:title>`);
  fs.writeFileSync(cp, c);
} catch (e) {}

console.log('package ready', OUT_DIR);
