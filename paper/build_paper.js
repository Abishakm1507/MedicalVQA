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
// ------------------------------------------------------------------ CONTENT
const title = 'Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images: Pipeline Implementation and Measured Results with General and Medical Vision-Language Models';

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

const abstractText = 'Vision-language models can answer questions about medical images fluently, but they may state findings that are not in the image, give no indication of where a finding is, and rarely express doubt. We built a pipeline that tries to answer only what it can point to. Numbered candidate regions are drawn on the image, the vision-language model must answer in a JSON contract that cites those regions, a verifier removes claims that cite no valid region, and a confidence score that combines answer agreement, semantic entropy and the localizer score decides whether to answer or abstain. We ran Qwen2.5-VL-3B and MedGemma-4B, both in 4-bit precision, on a seeded random subset of the SLAKE English test split (250 questions on 77 images) with the confidence weights and threshold fitted on a separate validation subset. Strict exact-match accuracy was 38 to 42 percent for all four model and track combinations, and grounding or verification did not change it significantly (Condition 2 versus greedy Condition 1, McNemar p = 0.37 for MedGemma and p = 1.0 for Qwen). The fused confidence ranked correct answers above wrong ones well (AUROC 0.79 to 0.83). With abstention, MedGemma-4B answered about half of the questions at 67 to 72 percent accuracy, while Qwen2.5-VL-3B answered only 22 percent of them at 75 to 80 percent accuracy. Qwen2.5-VL-3B also broke the JSON contract far more often, and 43 percent of its positive claims in the oracle-region track cited no valid region, against 9 percent for MedGemma-4B. The results come from a subset, strict scoring, one decoding setup and a single dataset, and they are not clinical validation.';

const body = [];
const push = (...x) => body.push(...x);

push(
  P('Abstract', '*Abstract*—' + abstractText),
  P('Keywords', '*Keywords*—medical visual question answering, vision-language models, visual grounding, uncertainty estimation, selective prediction, MedSAM'),

  H1('Introduction'),
  B('A radiologist who reads an answer from a vision-language model (VLM) has two questions. Is it right, and where in the image does it come from? A plain-text answer addresses neither. The model may name an abnormality that is not on the scan, it gives the reader no region to check, and it sounds just as sure when it is guessing.'),
  B('We want a system that follows one rule: it only claims what it can point to. Every finding must cite a numbered region of the image, a finding that cites no valid region is removed, and when the evidence is weak the system says “uncertain” and does not answer.'),
  B('This paper reports the implementation of that pipeline and the experiments we could run on it with free GPU resources. We are explicit about what was measured and what was not. The two models were run on a seeded random subset of the SLAKE test cohort, the confidence gate was fitted on a separate validation subset and then frozen, and the results include bootstrap intervals and paired tests. The localization stage was not evaluated on real detector output, and we say so wherever a result could be mistaken for it.'),
  B('The contributions are as follows. (i) A six-stage pipeline with a typed JSON contract between the VLM and the verifier, behind one client interface for Qwen2.5-VL-3B and MedGemma-4B. (ii) A runtime verifier that never uses ground truth, so that evaluation cannot leak into prediction. (iii) A confidence gate whose weights and threshold are fitted on validation images and then frozen, not constants. (iv) An image-disjoint cohort design with pilot images excluded from the test cohort. (v) A measured comparison of two models on two tracks and three conditions, with its limits stated.'),
  B('Related work is in Section II. The system and data are in Section III. Section IV gives results and their limits. Sections V and VI give the conclusion and future work.'),

  H1('Related Work'),
  H2('Medical VQA and Datasets'),
  B('SLAKE [1] provides English and Chinese questions on 642 images of several modalities, with pixel masks and bounding boxes for organs and lesions. Because it has region annotations, it is the main benchmark for checking whether a cited region is correct. VQA-RAD [2] contains clinician-written questions on radiology images but has no region annotations, so we do not use it here. Chest-radiograph detection sets such as RSNA Pneumonia [3] and VinDr-CXR [4] provide boxes for training and checking a detector.'),
  H2('Grounding and Visual Prompting'),
  B('The Segment Anything Model [5] segments an object from a box or point prompt, and MedSAM [6] adapts it to medical images. Set-of-Mark prompting [7] overlays numbered marks on an image so that a language model can refer to regions by number. We use the same device to make the cited regions explicit and checkable.'),
  H2('Uncertainty and Selective Prediction'),
  B('Kuhn et al. [8] estimate uncertainty in language models by sampling several answers, grouping those with the same meaning and computing the entropy over the groups. We use this semantic entropy together with agreement among samples. Calibration is measured by the expected calibration error [9], and abstention is judged by how accuracy changes as coverage is reduced [10]. Hallucination in generative models is surveyed in [11].'),
  B('It is important to say what the confidence score is. It measures how consistent the model is with itself and how strong the localizer evidence is. It is not a probability of being correct, and as Section IV shows, its ECE is poor even though it ranks answers well.'),

  H1('System Description'),
  H2('Overview'),
  B('The pipeline has six stages. (1) The image and question are read. (2) A detector proposes up to k candidate boxes with scores, and MedSAM converts each box into a mask. (3) The regions are drawn on a copy of the image as R1 to Rk. (4) The VLM answers in a JSON contract with an answer, a list of findings, and for each finding a region identifier, a status (present, absent or uncertain) and a confidence. N answers are sampled. (5) The verifier checks each finding against the region set and removes those without a valid region. (6) The gate combines three signals and either returns the majority answer or abstains. The pipeline is drawn in Fig. 1. In the experiments of this paper the regions come from the dataset’s ground-truth boxes (Track B) or are absent (Track C), so stages 2 and 3 were exercised with oracle boxes and the trained detector and MedSAM masks were not part of the measured results. Table I gives the status of each component.'),
  FIG('fig1_architecture.png', 850, 1007, 'The six-stage pipeline. A confidence at or above the threshold returns the answer with its regions, and a lower one abstains.'),
  TABLE('Implementation and Evaluation Status of System Components', [2750, 2050], ['Component', 'Status'], [
    ['Dataset loaders and leakage checks', 'Built; SLAKE checked on real data'],
    ['Faster R-CNN candidate detector', 'Built; not evaluated (no CXR weights)'],
    ['MedSAM adapter', 'Built; verified on one image; not in the runs'],
    ['Qwen2.5-VL-3B and MedGemma-4B clients', 'Built; run on all arms below'],
    ['Claim extraction and verification', 'Built, unit-tested, run on all arms'],
    ['Confidence gate and calibration', 'Fitted on validation subset; frozen'],
    ['Metrics, bootstrap, McNemar', 'Built; applied to 250-question subset'],
    ['Full 945-question test cohort', 'Not run (free GPU budget)'],
    ['Track A (detector on real CXR)', 'Not available on SLAKE'],
    ['Web interface, expert review', 'Specified, not evaluated'],
  ]),
  H2('Datasets and Cohorts'),
  B('All experiments use SLAKE. The English test split has 1,061 questions and the English validation split has 1,053. A 10-question pilot on 9 test images and a 15-image validation pilot are kept out of every later evaluation. Removing the pilot images leaves 945 eligible test questions on 87 images, of which 864 have region annotations and 81 have none. For calibration, the 162 validation questions on the pilot images are removed, which leaves 891 questions. They are divided 80/20 by image with a fixed seed, giving 719 questions on 65 images for fitting and 172 questions on 16 images for assessment. No image appears in both parts, and the test split is never used for fitting. Splitting by image matters because several questions share one image.'),
  B('The free GPU tier made the full cohorts too slow, so the model runs use a seeded random subset of each part: 200 fit questions, 80 assessment questions and 250 test questions. The subset is drawn from the questions that have region annotations, the same questions are used for Tracks B and C and for both models, and the 250 test questions lie on 77 images. The cohorts are shown in Fig. 2. All intervals below therefore describe this subset, and the full 945-question test cohort was not run.'),
  FIG('fig2_cohorts.png', 850, 960, 'Cohorts. Darker boxes are the subsets that were run through the models.'),
  H2('Localization and Candidate Regions'),
  B('The specification asks for a YOLO- or DETR-style detector. Ultralytics YOLOv8 is released under the AGPL-3.0 licence, which would conflict with the MIT licence of this project, so we chose a Torchvision Faster R-CNN [12]. MedSAM ViT-B weights were checked by size and MD5, and the adapter produced a valid mask from a box on one SLAKE image.'),
  B('One rule is fixed. If the detector finds nothing above its threshold, the region set is empty. The system never invents boxes, such as image quadrants, to fill the gap. Any positive finding on an empty region set is therefore ungrounded.'),
  H2('Structured Contract and Claim Verification'),
  B('The VLM must return JSON. Each finding has a status. A “present” finding must cite a region in R. An “absent” finding cites NONE and is handled separately, as are “uncertain” findings, so that negative claims do not distort the grounding metrics. The verifier works on structure only: it checks that a cited identifier exists, and it never reads ground-truth boxes. Ground truth enters only in the evaluation layer. The unsupported-claim rate (USR) is the fraction of positive claims that are unsupported, and it is reported before filtering (USR_raw) and after (USR_filtered).'),
  H2('Confidence Gate'),
  B('From N = 5 samples at temperature 0.7 the answers are grouped into semantic clusters. Let a be the share of samples in the largest cluster and p_k the share in cluster k. The normalized entropy is'),
  EQ('H_n = −Σ p_k ln p_k / ln N'),
  B('and the fused confidence is'),
  EQ('c = w1·a + w2·(1 − H_n) + w3·ρ'),
  B('with the weights non-negative and summing to one, and ρ the localizer score of the cited region. For an answer with no region, no localizer score exists and the other weights are rescaled instead of assigning an artificial value. The decision rule is'),
  EQ('ŷ = y_majority if c ≥ τ, else “uncertain”'),
  B('The weights and τ are chosen by grid search on the validation-fit questions of each model and track, assessed on the validation-assessment questions, and frozen before the test questions are scored. None is hard-coded.'),
  H2('Tracks and Conditions'),
  B('Track B uses the dataset’s ground-truth boxes as oracle regions, which tests the VLM and verifier without detector errors. Track C is ungrounded answering. Track A, with detector-proposed regions, was found ineligible on SLAKE and is not used. Condition 1 is a free-form answer with N = 1 and greedy decoding, and it was run for Track C only. Condition 2 is a grounded answer with verification and no abstention. Condition 3 adds abstention. Conditions 2 and 3 reuse the same generated samples, so only the abstention decision differs between them.'),
  H2('Implementation'),
  B('Both models are loaded in 4-bit precision through Hugging Face Transformers on a Kaggle T4 GPU (MedGemma-4B is google/medgemma-4b-it, a gated model). Each generation is written as an artifact, re-read after writing, and skipped on a restart if it is complete. Replies are capped at 256 new tokens. Heavy inference ran on the free GPU, and calibration, scoring and statistics ran on a laptop CPU.'),
  B('Some replies do not satisfy the strict JSON contract, because they are cut off at the token cap or use a value outside the allowed set for the status field. Dropping those questions would bias the results towards short, well-formed replies. We therefore salvage them offline with one fixed rule, applied identically to every model, track and split. A reply is usable if it contains an answer string. Only complete findings with a valid status and confidence are kept, and invalid or cut-off findings are dropped, never repaired. A question is kept if at least three of its five samples are usable. By this rule between 19 and 92 of the 250 test questions per arm needed salvage and none was excluded (Table IV).'),

  H1('Results and Discussion'),
  H2('Evaluation Protocol'),
  B('Answers are compared with the reference after the same normalization on both sides: whitespace, code fences, quotes, case and trailing punctuation are removed, and then an exact match is required. This is strict. A correct answer that is worded differently, such as a sentence in place of a single word, counts as wrong, so the absolute accuracies understate what a clinician would accept. Abstained questions count as wrong in overall accuracy, and selective accuracy is computed on answered questions only. Confidence intervals are 95 percent image-clustered bootstrap intervals with 2,000 replicates, and paired comparisons use a paired bootstrap and McNemar’s test. The McNemar test is at question level and is not adjusted for image clusters.'),
  H2('Accuracy and Conditions'),
  B('Table II gives the main results. Overall accuracy lies between 38 and 42 percent for every arm in Conditions 1 and 2, and the confidence intervals overlap (Fig. 3). MedGemma-4B is higher than Qwen2.5-VL-3B on closed questions in Track C (68 versus 53 percent) and lower on open questions (25 versus 27 percent), but the overall difference of 4.4 points has an interval that includes zero (−1.2 to 10.4). Grounding and verification did not change accuracy measurably. Condition 2 minus Condition 1 in Track C is +1.2 points for MedGemma-4B (McNemar p = 0.37) and −0.4 points for Qwen2.5-VL-3B (p = 1.0), and Track B minus Track C in Condition 2 is −3.2 and +2.4 points (p = 0.10 and 0.40). We therefore have no evidence that region grounding improves accuracy for either model on this subset.'),
  TABLE('Test Results on 250 SLAKE Questions (Strict Exact Match, 95% CI). MG = MedGemma-4B, Q = Qwen2.5-VL-3B, B and C = track, C1 to C3 = condition', [1000, 1400, 900, 500, 1100], ['Arm', 'Accuracy', 'Open / closed', 'Cov.', 'Selective acc.'], [
    ['MG B C2', '0.39 (.33–.45)', '.22 / .63', '1.00', '0.39'],
    ['MG B C3', '0.33 (.27–.39)', '.18 / .54', '0.49', '0.68 (.59–.77)'],
    ['MG C C1', '0.41 (.36–.47)', '.24 / .66', '1.00', '0.41'],
    ['MG C C2', '0.42 (.37–.48)', '.25 / .68', '1.00', '0.42'],
    ['MG C C3', '0.36 (.30–.41)', '.19 / .59', '0.50', '0.72 (.64–.79)'],
    ['Q B C2', '0.40 (.34–.47)', '.29 / .57', '1.00', '0.40'],
    ['Q B C3', '0.16 (.12–.21)', '.10 / .25', '0.22', '0.75 (.65–.84)'],
    ['Q C C1', '0.38 (.31–.45)', '.30 / .51', '1.00', '0.38'],
    ['Q C C2', '0.38 (.31–.45)', '.27 / .53', '1.00', '0.38'],
    ['Q C C3', '0.17 (.13–.22)', '.12 / .25', '0.22', '0.80 (.67–.90)'],
  ]),
  FIG('fig3_accuracy.png', 968, 731, 'Test accuracy by arm. Conditions 1 and 2 show overall accuracy, and Condition 3 shows accuracy on answered questions only. Bars show 95 percent image-clustered intervals.'),
  H2('Abstention and Calibration'),
  B('Abstention is the one component that changed the answers. Because abstained questions count as wrong, overall accuracy falls (MedGemma-4B by 6.8 points in Track C, Qwen2.5-VL-3B by 20.8, both with McNemar p below 0.001), and that fall is built into the definition. The useful quantity is the accuracy on the answered questions. MedGemma-4B answered about half of the questions (coverage 0.49 and 0.50) at 67.5 and 71.8 percent accuracy, compared with 39 and 42 percent when it always answered. Qwen2.5-VL-3B answered only 22 percent of the questions, at 74.5 and 79.6 percent. The accuracy–coverage curves are in Fig. 4. These gains come from questions that the model is consistent about being answered more often correctly, and they cost coverage: for Qwen2.5-VL-3B the gate withholds four of every five questions.'),
  FIG('fig4_risk_coverage.png', 974, 767, 'Accuracy on answered questions against coverage as the threshold is varied, on the test subset.'),
  TABLE('Confidence Quality and Grounding Measures on the Test Subset (Condition 2)', [1000, 1300, 500, 600, 500, 600], ['Arm', 'AUROC (CI)', 'ECE', 'Agree.', 'USR', 'Point.'], [
    ['MG B', '0.83 (.78–.87)', '0.59', '0.73', '0.09', '0.88'],
    ['MG C', '0.83 (.78–.87)', '0.39', '0.75', '1.00', '0.00'],
    ['Q B', '0.80 (.75–.85)', '0.57', '0.62', '0.43', '0.70'],
    ['Q C', '0.79 (.74–.85)', '0.36', '0.61', '1.00', '0.00'],
  ]),
  B('The fused confidence separates correct from wrong answers well (Table III): the AUROC under Condition 2 is 0.79 to 0.83 with intervals that exclude 0.5 by a wide margin. It is not calibrated as a probability, however. ECE is 0.36 to 0.59 under Condition 2, and in Condition 3 it falls in Track B (0.33 and 0.25) but rises in Track C (0.67 and 0.75), because after thresholding the retained answers are far more accurate than the confidence value suggests. Confidence works as a ranking score for deciding when to abstain, and should not be read as a probability of being correct. The AUROC values reported under Condition 3 (0.88 to 0.97) are computed by the evaluator on that condition’s output and are not comparable with the Condition 2 values, so we do not claim that abstention improves the ranking.'),
  B('The fitted weights are informative about what the gate learned. On every arm the grid search selected w1 = 0, w2 = 0.05 and w3 = 0.95, that is, no weight on answer agreement. In Track B the localizer score ρ is constant, because the oracle boxes carry no detector score, and in Track C there is no region, so ρ is absent and the other weights are rescaled. In both tracks the gate therefore reduces in effect to a threshold on the semantic entropy, and the learned thresholds (0.99 in Track B and 0.04 in Track C) are consistent with that. This is our reading of the design and fitted values, and we did not run a separate ablation of the individual signals. The localizer term could not be assessed at all with oracle boxes, so the claim in the design that a localizer score improves confidence remains untested.'),
  H2('Grounding Measures'),
  B('In Track B, USR_raw is the share of positive claims that cite no valid region. It is 0.09 for MedGemma-4B and 0.43 for Qwen2.5-VL-3B, which means that nearly half of Qwen2.5-VL-3B’s positive claims could not be tied to a candidate region, against about one in eleven for MedGemma-4B. Pointing accuracy is 0.88 for MedGemma-4B and 0.70 for Qwen2.5-VL-3B. Two caveats apply. Pointing accuracy in Track B is scored against the same oracle boxes the model was shown, so it measures whether the model picks one of the offered boxes correctly and not how well a localizer works. The evaluator reports USR_filtered equal to USR_raw, and we have not investigated why filtering leaves the rate unchanged, so we report only the raw rate. In Track C there are no regions, so USR is 1.0 and pointing accuracy is 0.0 by construction, and neither number is a finding.'),
  H2('Contract Adherence and Salvage'),
  B('The two models differ sharply in how well they follow the JSON contract. Between 6.0 and 7.6 percent (19 of 250 for Track B and 15 of 250 for Track C) of MedGemma-4B’s test questions needed salvage, against 34 percent (86 of 250) and 37 percent (92 of 250) for Qwen2.5-VL-3B. Qwen2.5-VL-3B often copied the schema text literally, for example the status “present|absent|uncertain”, or used values such as “unknown”. Salvage dropped no more than one sample in any arm, and no question was excluded, so all 250 questions are scored in every arm. Because salvage keeps the answer but drops invalid findings, Qwen2.5-VL-3B’s grounding measures rest on fewer valid findings than MedGemma-4B’s, and its much lower coverage under abstention may partly reflect noisier sampling. We did not test a strict-only analysis.'),
  TABLE('Questions Needing Offline Salvage on the 250 Test Questions', [3000, 1800], ['Arm', 'Questions salvaged'], [
    ['MedGemma-4B, Track B', '19 (7.6%)'],
    ['MedGemma-4B, Track C', '15 (6.0%)'],
    ['Qwen2.5-VL-3B, Track B', '86 (34.4%)'],
    ['Qwen2.5-VL-3B, Track C', '92 (36.8%)'],
  ]),
  H2('Earlier Pilot and Corrections Made During the Work'),
  B('An earlier 10-question pilot (6 correct for Qwen2.5-VL-3B and 7 for MedGemma-4B) used different settings and is superseded by the results above. During the work several errors were found by audit and corrected, and we list them because they could easily have gone unnoticed. The cohort script looked for annotation files at the wrong path, so earlier reports counted 1,051 eligible questions, and the correct count is 945. A calibration result obtained on synthetic data had been written up as if it came from SLAKE. The evaluation script had used placeholder confidence, agreement and pointing values derived from correctness, which made AUROC circular, and it was rewritten to compute them from real samples. In the generation code, an end-of-sequence override made MedGemma-4B run to the token cap on every answer, the greedy condition passed a temperature of zero that the library rejects, and half-precision (fp16) on the T4 produced invalid probabilities, so bfloat16 compute was used. All were fixed before the reported runs, and the greedy fix was confirmed on real hardware by the Condition 1 run.'),
  H2('Software Verification'),
  B('The test suite ran in a clean Python 3.11 environment, with the generation clients, the salvage rule and the driver covered by their own unit tests. Passing tests show that the code does what it was written to do. They do not show that the pipeline is accurate, and the accuracy statements above rest on the measured runs.'),
  H2('Limitations'),
  B('The results come from a random subset of 250 test questions on 77 images, and the intervals are wide. The full 945-question cohort was not run. Accuracy is strict exact match, so it understates correct answers that are worded differently. Both models were run in 4-bit precision with a 256-token cap and one sampling setup, and conclusions may change with other settings. Track B uses oracle boxes, so the detector and MedSAM stages did not contribute to the measured results, and Track A, which would test the detector on real chest radiographs, is not available on SLAKE. The localizer term of the gate is untested, and the fitted gate acts as an entropy threshold. Track C values of USR and pointing are fixed by construction. Qwen2.5-VL-3B’s weaker contract adherence confounds the model comparison. The McNemar test is not adjusted for image clusters. SLAKE is a curated benchmark, so performance on routine clinical images should be expected to be worse. Perturbation robustness, masked-region ablation and blinded expert review were specified but not carried out. Nothing here is clinical validation, and the system is not suitable for clinical use.'),

  H1('Conclusion'),
  B('We implemented a pipeline in which a vision-language model must cite numbered image regions, unsupported claims are removed, and a fused confidence score decides whether to answer or abstain, and we measured it with Qwen2.5-VL-3B and MedGemma-4B on a 250-question SLAKE test subset. Grounding and verification did not change accuracy measurably, which stayed at 38 to 42 percent under strict scoring. The confidence score ranked correct answers well, and abstention raised the accuracy of answered questions to 67 to 80 percent, at a coverage of 49 to 50 percent for MedGemma-4B and 22 percent for Qwen2.5-VL-3B. MedGemma-4B kept its claims grounded far more consistently than Qwen2.5-VL-3B. Whether region grounding improves medical VQA with real detector regions, and on the full cohort, is still open.'),

  H1('Future Work'),
  B('The next steps are in order. First, run the full 945-question test cohort and the full calibration cohort with the frozen method. Second, train and evaluate a detector on RSNA Pneumonia and VinDr-CXR so that Track A and the localizer term can be tested, and ablate the gate signals separately. Third, add a looser answer-matching measure as a clearly labelled secondary result and a strict-only analysis that excludes salvaged questions. Fourth, run the perturbation and masked-region ablation suites. Fifth, add VQA-RAD as an ungrounded baseline and run the blinded expert review. Last, finish and test the web interface.'),

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
