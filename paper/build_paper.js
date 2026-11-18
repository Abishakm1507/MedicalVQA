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

const abstractText = 'While vision-language models fluently answer questions about medical images, they often state findings that are not in the image, give no information about where a finding is, and rarely express doubt. We created a pipeline so that the model answers only what it can point to. The model is given an image with numbered candidate regions and must answer in a JSON format that cites those regions. A verifier removes claims that cite no valid region; it checks that the cited region exists, not that it supports the finding. The system then answers or abstains based on a confidence score that combines answer agreement, semantic entropy and the localizer score. We tested Qwen2.5-VL-3B and MedGemma-4B in 4-bit precision on a seeded random subset of the SLAKE English test split (250 questions on 77 images), with the confidence weights and threshold fitted on a separate validation subset. Strict exact-match accuracy was 38 to 42 percent for all four model and track combinations, and grounding or verification did not change it significantly (Condition 2 versus greedy Condition 1, McNemar p = 0.37 for MedGemma and p = 1.0 for Qwen). The fused confidence ranked correct answers above wrong answers well, with an AUROC of 0.79 to 0.83. With abstention, MedGemma-4B answered about half of the questions at 67 to 72 percent accuracy, while Qwen2.5-VL-3B answered only 22 percent of the questions, at 75 to 80 percent accuracy. Qwen2.5-VL-3B also broke the JSON contract far more often, and 43 percent of its positive claims in the oracle-region track cited no valid region, against 9 percent for MedGemma-4B. A strict-only re-analysis and a signal ablation show that the conclusions hold without the salvaged replies and that the fused score adds nothing over agreement or entropy alone. These results come from a small subset with oracle regions and do not represent clinical validation.';

const body = [];
const push = (...x) => body.push(...x);

push(
  P('Abstract', '*Abstract*—' + abstractText),
  P('Keywords', '*Keywords*—medical visual question answering, vision-language models, visual grounding, uncertainty estimation, selective prediction, MedSAM'),

  H1('Introduction'),
  B('A radiologist has two questions when a vision-language model (VLM) responds to a question. The first is whether the VLM answered correctly. The second is where in the image the answer comes from. Plain text answers neither. The VLM may respond with an abnormality that does not appear in the image, give the reader no location to check, and still sound just as certain when it is guessing.'),
  B('Our system needs to follow one rule: it only makes claims that the evidence supports. Any finding that cites no valid region is removed. If the evidence is weak, the system should respond “uncertain” and not give an answer.'),
  B('We report the development of such a system and the experiments we could run with the limited free GPU time available. We state clearly what was measured and what was not. The two VLMs were run on a random subset of the SLAKE test split, the confidence gate was fitted on a separate validation subset and then kept fixed, and the results include bootstrap intervals and paired tests. The localization step was not tested with real detector output, and we say so wherever a result could be mistaken for one.'),
  B('There are five contributions. (i) A six-stage pipeline with a typed JSON contract between the VLM and the verifier, behind one client interface for Qwen2.5-VL-3B and MedGemma-4B. (ii) A runtime verifier that never uses ground truth, so that evaluation and prediction are strictly separated. (iii) A confidence gate whose weights and threshold are fitted on validation images and then frozen. (iv) An image-disjoint cohort design in which pilot images are not included in the test cohort. (v) A comparison of two models on two tracks and three conditions, with its limits stated.'),
  B('Related work is in Section II. The system and data are described in Section III. Results and their limits are in Section IV. The conclusion and future work are in Sections V and VI, respectively.'),

  H1('Related Work'),
  H2('Medical VQA and Datasets'),
  B('SLAKE [1] contains 642 images across several modalities, with questions in English and Chinese, pixel masks, and bounding boxes of organs and lesions. Because of these region annotations, it is the main benchmark for checking whether a cited region is correct. VQA-RAD [2] contains questions about radiology images but has no region annotations, so we do not use it here. Chest-radiograph detection sets such as RSNA Pneumonia [3] and VinDr-CXR [4] provide boxes for training and validating a detector.'),
  H2('Vision-Language Models'),
  B('General-purpose VLMs such as Qwen2.5-VL [13] and medical adaptations such as LLaVA-Med [14] and MedGemma [15] can answer medical questions, but they are rarely evaluated for whether their answers can be traced to image regions or whether they know when not to answer. We use Qwen2.5-VL-3B and MedGemma-4B as a general and a medical model.'),
  H2('Grounding and Visual Prompting'),
  B('The Segment Anything Model [5] segments objects from box or point prompts, and MedSAM [6] extended it to medical images. Set-of-Mark prompting [7] places numbered marks on an image so that regions can be referred to by a language model. We take the same approach to make the cited regions explicit and checkable.'),
  H2('Uncertainty and Selective Prediction'),
  B('Kuhn et al. [8] estimate uncertainty in language models by sampling multiple responses, clustering those that are semantically equivalent, and computing the entropy over the clusters. In addition, we include agreement among the samples. Calibration is assessed by the expected calibration error [9], and abstention is evaluated by how answer accuracy changes as coverage is reduced [10]. Hallucination in generative models is surveyed in [11].'),
  B('We should be clear about what the confidence score is. It shows how consistent the model is with itself and how strong the localizer evidence is. It should not be interpreted as the probability of being correct. As Section IV shows, its ECE is poor even though it ranks answers well.'),

  H1('System Description'),
  H2('Overview'),
  B('The pipeline has six stages. (1) The image and question are read. (2) A detector proposes up to k candidate boxes with scores, and MedSAM converts each box into a mask. (3) The regions are drawn on a copy of the input image as R1 to Rk. (4) The VLM returns an answer and a set of findings in a JSON contract. Each finding has a region identifier, a status (present, absent or uncertain) and a confidence. N answers are sampled. (5) The verifier checks each finding against the region set and removes those without a valid region. (6) The gate combines three signals and returns the majority answer if the confidence reaches the threshold, and abstains otherwise. The pipeline is shown in Fig. 1. In the experiments reported here, the regions are taken from the ground-truth boxes of the dataset (Track B) or are absent (Track C). Stages 2 and 3 were therefore exercised with oracle boxes, and the outputs of the trained detector and MedSAM masks are not part of the measured results. Table I shows the status of each component.'),
  FIG('fig1_architecture.png', 850, 1007, 'The six-stage pipeline. A confidence at or above the threshold returns the answer with its regions, and a lower one abstains.'),
  TABLE('Implementation and Evaluation Status of System Components', [2750, 2050], ['Component', 'Status'], [
    ['Dataset loaders and leakage checks', 'Built; SLAKE checked on real data'],
    ['Faster R-CNN candidate detector', 'Built; not evaluated (no CXR weights)'],
    ['MedSAM adapter', 'Built; verified on one image; not in runs'],
    ['Qwen2.5-VL-3B and MedGemma-4B clients', 'Built; run on all arms below'],
    ['Claim extraction and verification', 'Built, unit-tested, run on all arms'],
    ['Confidence gate and calibration', 'Fitted on validation subset; frozen'],
    ['Metrics, bootstrap and McNemar test', 'Built; applied to 250-question subset'],
    ['Full 945-question test cohort', 'Not run (free GPU budget)'],
    ['Track A (detector on real CXR)', 'Not available on SLAKE'],
    ['Web interface, expert review', 'Specified, not evaluated'],
  ]),
  H2('Datasets and Cohorts'),
  B('All experiments use SLAKE. The English test split contains 1,061 questions and the English validation split contains 1,053. A 10-question pilot on 9 test images and a 15-image validation pilot are excluded from all later evaluation. Removing the pilot images leaves 945 test questions on 87 images, of which 864 have region annotations and 81 do not. For calibration, the 162 validation questions on the pilot images are removed, leaving 891 questions. These are divided 80/20 by image using a fixed seed, giving 719 questions on 65 images for fitting and 172 questions on 16 images for assessment. No image appears in both the fit and assessment sets, and the test set is never used for fitting. Splitting by image is important because several questions share the same image.'),
  B('The free GPU tier made the full cohorts too slow, so the model runs use a seeded random subset of 200 fit questions, 80 assessment questions and 250 test questions. The subset is drawn from the questions that have region annotations, the same questions are used for both tracks and both models, and the 250 test questions lie on 77 images. The cohorts are illustrated in Fig. 2. All intervals below describe this subset, and the full 945-question test cohort was not run.'),
  FIG('fig2_cohorts.png', 850, 960, 'Cohorts. Darker boxes are the subsets that were run through the models.'),
  H2('Localization and Candidate Regions'),
  B('The specification asks for a YOLO- or DETR-style detector. The AGPL-3.0 licence of Ultralytics YOLOv8 conflicts with the MIT licence of our project, so we chose a Torchvision Faster R-CNN [12] instead. The MedSAM ViT-B weights were verified by size and MD5, and the adapter generated a valid mask from a box on one SLAKE image.'),
  B('One rule is fixed. The system never invents boxes to fill a gap: if the detector finds nothing above its threshold, the region set is empty. Any positive finding on an empty region set is ungrounded.'),
  H2('Structured Contract and Claim Verification'),
  B('The VLM returns JSON. Each finding has a status. A finding with status “present” must cite a region in R. A finding with status “absent” cites NONE and is handled separately, as is a finding with status “uncertain,” so that negative claims do not affect the grounding metrics. The verifier is a structural check: it confirms that a cited identifier exists in the region set and never reads ground-truth boxes. It does not establish that the cited region actually contains the finding, so “backed by a region” in this paper means that a valid region is cited, and semantic support remains unverified. Ground truth enters only in the evaluation layer. There, the unsupported-claim rate (USR) is the fraction of positive claims that are unsupported, calculated before filtering (USR_raw) and after (USR_filtered), and pointing accuracy is the fraction of questions for which at least one cited region has an intersection-over-union (IoU) of at least 0.5 with an annotated box.'),
  H2('Confidence Gate'),
  B('For each question, the answers from N = 5 samples drawn at temperature 0.7 (fixed by the design and not tuned) are grouped into clusters. In the reported runs, two answers fall in the same cluster only if they are identical after lower-casing and whitespace normalization. An embedding-based clusterer is implemented but was not used, so paraphrases count as different answers and the entropy may be overestimated. Let a be the fraction of samples in the largest cluster and p_k the fraction of samples in cluster k. The normalized entropy is defined as'),
  EQ('H_n = −Σ p_k ln p_k / ln N'),
  B('and the fused confidence is'),
  EQ('c = w1·a + w2·(1 − H_n) + w3·ρ'),
  B('where the weights are non-negative and sum to one, and ρ is the localizer score of the cited region. In Track B every oracle box is assigned ρ = 1, so ρ is constant. In Track C the positive claims cite no region, so they are ungrounded and ρ contributes 0 without rescaling. Only absent or uncertain claims omit ρ and rescale the other weights. The decision rule is'),
  EQ('ŷ = y_majority if c ≥ τ, else “uncertain”'),
  B('The weights are searched on a grid with step 0.05 over the simplex, and τ from 0 to 1 in steps of 0.01. The objective is to minimize the selective risk (the error rate on answered questions) on the validation-fit questions of each model and track, subject to a coverage of at least 0.10, which prevents abstaining on everything. AUROC and ECE are diagnostics only. The selected values are assessed on the validation-assessment questions and frozen before the test questions are scored, and none is hard-coded. ECE uses 10 equal-width bins over the answered questions.'),
  H2('Tracks and Conditions'),
  B('Track B uses ground-truth bounding boxes as oracle regions, which tests the VLM and verifier without detector errors. Track C is ungrounded answering. Track A, which uses detector-proposed regions, was found ineligible on SLAKE and is not used. Condition 1 is a free-form answer with N = 1 and greedy decoding, and it was run for Track C only. Condition 2 is a grounded answer with verification and no abstention. Condition 3 adds abstention. Conditions 2 and 3 reuse the same generated samples, so only the abstention decision differs between them.'),
  H2('Implementation'),
  B('Both models were run in 4-bit precision using Hugging Face Transformers on a Kaggle T4 GPU (MedGemma-4B is google/medgemma-4b-it, a gated model). Each generation is written as an artifact, re-read after writing, and skipped on a restart if it is complete. We capped replies at 256 new tokens. Heavy inference was done on the free GPU, and calibration, scoring and statistics were done on a laptop CPU.'),
  B('Some replies do not strictly follow the JSON contract, because they are cut off at the token limit or use a value outside the allowed set for the status field. Removing these questions would bias the results towards short, well-formed replies. We therefore salvaged them offline, using one fixed rule that is applied in the same way to all models, tracks and splits. A reply is usable if it contains an answer string. Only complete findings with a valid status and confidence are kept, and invalid or cut-off findings are dropped, never repaired. A question is kept if at least 3 of its 5 samples are usable, and its unusable samples are dropped. By this rule, between 19 and 92 of the 250 test questions per arm needed salvage and none was excluded (Table V).'),

  H1('Results and Discussion'),
  H2('Evaluation Protocol'),
  B('Answers and references are normalized before comparison: whitespace, code fences, quotes, case and trailing punctuation are removed. After normalization, an exact match is required. This strict comparison means that the absolute accuracies understate what a clinician would consider correct. Abstained questions are counted as wrong in overall accuracy, and selective accuracy is computed on the answered questions only. Confidence intervals are 95 percent image-clustered bootstrap intervals with 2,000 replicates, and paired comparisons use a paired bootstrap and McNemar’s test. The McNemar test is performed at the question level and is not adjusted for image clusters.'),
  H2('Accuracy and Conditions'),
  B('The main results are in Table II. In Conditions 1 and 2, accuracy lies between 38 and 42 percent for all arms, and the confidence intervals overlap (Fig. 3). Compared with Qwen2.5-VL-3B, MedGemma-4B scores higher on closed questions in Track C (68 versus 53 percent) and lower on open questions (25 versus 27 percent). The overall difference of 4.4 points has a confidence interval of −1.2 to 10.4, which includes zero. Grounding and verification did not appreciably change accuracy. In Track C, Condition 2 minus Condition 1 is +1.2 points for MedGemma-4B (p = 0.37) and −0.4 points for Qwen2.5-VL-3B (p = 1.0). In Condition 2, Track B minus Track C is −3.2 points for MedGemma-4B (p = 0.10) and +2.4 points for Qwen2.5-VL-3B (p = 0.40). We therefore have no evidence that region grounding improves accuracy for either model on this subset.'),
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
  B('Abstention was the only component that changed the answers. Since abstained questions count as wrong, overall accuracy falls (by 6.8 points for MedGemma-4B in Track C and by 20.8 for Qwen2.5-VL-3B, both with McNemar p < 0.001), and that fall is built into the definition. The more useful quantity is the accuracy on the answered questions. MedGemma-4B answered about half of the questions (coverage 0.49 and 0.50) at 67.5 and 71.8 percent accuracy, compared with 39 and 42 percent when it always answered. Qwen2.5-VL-3B answered only 22 percent of the questions, at 74.5 and 79.6 percent accuracy. The curves of accuracy against coverage are in Fig. 4. These gains come from the questions on which the model is consistent being answered, and they cost coverage: for Qwen2.5-VL-3B the gate withholds four of every five questions.'),
  FIG('fig4_risk_coverage.png', 974, 767, 'Accuracy on answered questions against coverage as the threshold is varied, on the test subset.'),
  TABLE('Confidence Quality and Grounding Measures on the Test Subset (Condition 2)', [1000, 1300, 500, 600, 500, 600], ['Arm', 'AUROC (CI)', 'ECE', 'Agree.', 'USR', 'Point.'], [
    ['MG B', '0.83 (.78–.87)', '0.59', '0.73', '0.09', '0.88'],
    ['MG C', '0.83 (.78–.87)', '0.39', '0.75', '1.00', '0.00'],
    ['Q B', '0.80 (.75–.85)', '0.57', '0.62', '0.43', '0.70'],
    ['Q C', '0.79 (.74–.85)', '0.36', '0.61', '1.00', '0.00'],
  ]),
  B('The fused confidence separates correct from wrong answers well (Table III). The AUROC in Condition 2 is between 0.79 and 0.83, and its intervals exclude 0.5 by a wide margin. It is, however, not calibrated as a probability. ECE in Condition 2 is between 0.36 and 0.59. In Condition 3 it falls in Track B (0.33 and 0.25) but rises in Track C (0.67 and 0.75), because after thresholding the retained answers are far more accurate than the confidence value suggests. Confidence is useful as a ranking score for deciding when to abstain, and it should not be read as a probability of being correct. An ECE of this size means that the confidence values are not calibrated, and we do not claim that they are. We do not report AUROC for Condition 3. Its records label abstained questions as wrong, so an AUROC computed on them would be circular and inflated, and the Condition 2 values are the valid ones.'),
  B('The fitted weights show what the gate learned. In all arms, the grid search chose w1 = 0, w2 = 0.05 and w3 = 0.95, that is, no weight on answer agreement and almost all weight on ρ. Because ρ is constant (1 in Track B, 0 in Track C), the confidence varies between questions only through the term 0.05 (1 − H_n), so the gate is in effect a threshold on semantic entropy. The learned thresholds are consistent with this: in Track C, c = 0.05 (1 − H_n) and τ = 0.04 keeps the questions with H_n ≤ 0.2, and in Track B, c = 0.95 + 0.05 (1 − H_n) and τ = 0.99 does the same. To check how much each signal matters, we ranked the test questions by agreement alone, by 1 − H_n alone and by the fused score, using Condition 2 correctness. The AUROC values are 0.83, 0.83, 0.80 and 0.80 for agreement, 0.83, 0.83, 0.80 and 0.79 for 1 − H_n, and 0.83, 0.83, 0.80 and 0.79 for the fused score (MG B, MG C, Q B, Q C), and the accuracy at the gate’s coverage is identical, because on five samples agreement and entropy rank questions almost identically. The fused score therefore adds nothing over either signal in our data, which also explains why the grid search had no basis to prefer agreement. The localizer term could not be assessed with oracle boxes, so the claim in the design that a localizer score improves confidence remains untested.'),
  H2('Calibration Split and Paired Comparisons'),
  B('The same gate behaved as follows on the validation subsets. On the 200 fit questions the gate answered 19 to 47 percent of questions, and on the 80 assessment questions it answered 25 to 48 percent. Accuracy on the answered assessment questions was 0.84 to 0.95, which is higher than the 0.68 to 0.80 seen on the test subset. The assessment sets are small, so only about 20 to 40 questions were answered in each arm, and the threshold is tuned on the fit questions. We therefore treat the test values as the honest estimate and the assessment values as optimistic and noisy.'),
  B('Pairwise comparisons are given in Table IV. The differences between models and between tracks are small relative to their intervals, and none of them reaches significance in the paired bootstrap, so we do not rank the two models on accuracy. The one clear effect is the loss of overall accuracy from abstention, which follows from counting abstained questions as wrong. Because McNemar’s test treats questions as independent although several questions share an image, its p-values are somewhat optimistic, and we rely mainly on the clustered intervals.'),
  TABLE('Paired Accuracy Differences on the 250 Test Questions', [2300, 700, 1200, 700], ['Comparison', 'Diff.', '95% CI', 'p'], [
    ['MG C: Cond 2 − Cond 1', '+0.012', '−.004 to .030', '0.37'],
    ['Q C: Cond 2 − Cond 1', '−0.004', '−.043 to .038', '1.00'],
    ['MG C: Cond 3 − Cond 2', '−0.068', '−.098 to −.041', '<0.001'],
    ['Q C: Cond 3 − Cond 2', '−0.208', '−.272 to −.150', '<0.001'],
    ['MG: Track B − Track C', '−0.032', '−.071 to .004', '0.10'],
    ['Q: Track B − Track C', '+0.024', '−.020 to .065', '0.40'],
    ['Track C: MG − Q', '+0.044', '−.012 to .104', '0.14'],
  ]),
  H2('Grounding Measures'),
  B('In Track B, USR_raw is the fraction of positive claims that cite no valid region. It is 0.09 for MedGemma-4B and 0.43 for Qwen2.5-VL-3B, which means that about 43 percent of Qwen2.5-VL-3B’s positive claims could not be tied to a candidate region, against about 9 percent for MedGemma-4B. Pointing accuracy is 0.88 for MedGemma-4B and 0.70 for Qwen2.5-VL-3B. Two caveats apply. In Track B, pointing accuracy is scored against the same oracle boxes the model was shown, so it measures whether the model picks one of the offered boxes and not how well a localizer works. The evaluator also reports USR_filtered equal to USR_raw, and we have not investigated why, so we report only the raw rate. In Track C there are no regions, so USR is 1.0 and pointing accuracy is 0.0 by construction, and neither number is a finding.'),
  H2('Contract Adherence and Salvage'),
  B('The two models differ sharply in how well they follow the JSON contract. Between 6.0 and 7.6 percent of MedGemma-4B’s test questions needed salvage (19 of 250 for Track B and 15 of 250 for Track C), against 34 percent (86 of 250) and 37 percent (92 of 250) for Qwen2.5-VL-3B. Qwen2.5-VL-3B often copied the schema text literally, for example the status “present|absent|uncertain”, or used values such as “unknown”. Salvage dropped no more than one sample in any arm, and no question was excluded, so all 250 questions are scored in every arm. Because salvage keeps the answer but drops invalid findings, Qwen2.5-VL-3B’s grounding measures rest on fewer valid findings than MedGemma-4B’s, and its much lower coverage under abstention may partly reflect noisier sampling. Because following the contract is itself an outcome, we repeated the evaluation on the questions whose replies satisfied the contract with no salvage (Table V), using the same frozen weights and thresholds. Accuracy is 1 to 5 points higher on these questions in every arm, probably because the salvaged questions tend to have longer and harder answers, and none of the conclusions changes. Conditions 1 and 2 still do not differ, the accuracy on answered questions is still 0.68 to 0.84, and Qwen2.5-VL-3B still answers about a quarter of the questions against half for MedGemma-4B. The strict-only sets are not random subsets, and for Qwen2.5-VL-3B they hold only 158 to 164 of the 250 questions, so these numbers are a robustness check and not a replacement for the main results.'),
  TABLE('Salvaged Versus Strict-Only Evaluation on the Test Subset (Same Frozen Gate)', [800, 1000, 900, 900, 1200], ['Arm', 'Strict n (salvaged)', 'Cond 2 acc. all', 'Cond 2 acc. strict', 'Cond 3 sel. acc. all / strict'], [
    ['MG B', '231 (19)', '0.39', '0.40', '0.68 / 0.68'],
    ['MG C', '235 (15)', '0.42', '0.45', '0.72 / 0.73'],
    ['Q B', '164 (86)', '0.40', '0.45', '0.75 / 0.84'],
    ['Q C', '158 (92)', '0.38', '0.43', '0.80 / 0.75'],
  ]),
  H2('Earlier Pilot and Corrections Made During the Work'),
  B('An earlier 10-question pilot (6 correct for Qwen2.5-VL-3B and 7 for MedGemma-4B) used different settings and is superseded by the results above. Audits during the work found and corrected several errors: the cohort script looked for annotation files in the wrong directory (earlier reports counted 1,051 eligible questions instead of 945), a calibration result on synthetic data had been written up as if it came from SLAKE, and the evaluation script had used placeholder confidence, agreement and pointing values derived from correctness, which made the AUROC circular. In the generation code, the greedy condition passed a temperature of 0 that the library rejects, and fp16 on the T4 GPU produced invalid probabilities, so bfloat16 compute was used. All were fixed before the reported runs.'),
  H2('Software Verification'),
  B('The generation clients, the salvage rule and the driver have their own unit tests, and these passed. Passing tests show that the code performs as intended, not that the pipeline is accurate; the accuracy statements above rest on the measured runs.'),
  H2('Limitations'),
  B('The results come from a random subset of 250 test questions on 77 images, so the intervals are wide, and the full 945-question cohort was not run. Accuracy is strict exact match, and it may therefore understate correct answers that are worded differently. Both models were run in 4-bit precision with a 256-token cap and one sampling setup, and the results might differ with other settings. Track B uses oracle boxes, so the detector and MedSAM stages did not contribute to the measured results, and Track A, which would test the detector on real chest radiographs, is not available on SLAKE. The localizer component of the gate is untested, and in this study the fitted gate acts as an entropy threshold. The verifier checks only that a cited region exists, so whether a region supports its finding is not evaluated. Semantic clustering is exact match after normalization, and the fused gate adds nothing over a single signal. The USR and pointing values in Track C are fixed by construction. The weaker contract adherence of Qwen2.5-VL-3B makes the model comparison difficult. The McNemar test was performed without adjusting for image clusters. SLAKE is a curated benchmark, so performance on routine clinical images should be expected to be lower. Perturbation robustness, masked-region ablation and blinded expert review were specified but not carried out. Nothing in this study is clinical validation, and the system is not ready for clinical use.'),

  H1('Conclusion'),
  B('We built a pipeline in which an image carries numbered regions, unsupported claims are removed, and a fused confidence score decides whether to answer or abstain, and we measured it with Qwen2.5-VL-3B and MedGemma-4B on a 250-question SLAKE test subset. Grounding and verification did not improve accuracy, which stayed between 38 and 42 percent under strict scoring. The confidence score ranked correct answers above wrong ones well, and abstention raised the accuracy of answered questions to between 67 and 80 percent, at a coverage of 49 to 50 percent for MedGemma-4B and 22 percent for Qwen2.5-VL-3B. MedGemma-4B kept its claims grounded far more consistently than Qwen2.5-VL-3B. Whether region grounding improves medical VQA with real detector regions, whether a cited region really supports its finding, and whether the results hold on the full cohort, are still open questions.'),

  H1('Future Work'),
  B('The next steps are as follows. First, run the full 945-question test cohort and the full calibration cohort using the frozen method. Second, train and evaluate detectors on RSNA Pneumonia and VinDr-CXR to test Track A and the localizer term, and add an evaluation of whether a cited region really supports its finding, for example against annotated findings or by expert review. Third, add a looser answer-matching measure as a clearly labelled secondary result, and an embedding-based semantic clustering. Fourth, run the perturbation and masked-region ablation suites. Fifth, add VQA-RAD as an ungrounded baseline and run the blinded expert review. Last, complete and test the web interface.'),

  H5('Acknowledgment'),
  B('The authors are thankful to the Department of Artificial Intelligence and Data Science at R.M.K. Engineering College for their constant support during this project. The authors are also thankful to Mr. Sathesh Abraham Leo E, Assistant Professor, for his continuous mentorship.'),
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
  REF('S. Bai et al., “Qwen2.5-VL technical report,” arXiv:2502.13923, 2025.'),
  REF('C. Li et al., “LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day,” arXiv:2306.00890, 2023.'),
  REF('A. Sellergren et al., “MedGemma technical report,” arXiv:2507.05201, 2025.'),
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
