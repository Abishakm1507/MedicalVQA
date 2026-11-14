Add-Type -AssemblyName System.Drawing
$out = Join-Path $PSScriptRoot 'figs'
New-Item -ItemType Directory -Force $out | Out-Null

function New-Canvas($w, $h) {
  $bmp = New-Object System.Drawing.Bitmap $w, $h
  $bmp.SetResolution(300, 300)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.SmoothingMode = 'AntiAlias'; $g.TextRenderingHint = 'AntiAliasGridFit'
  $g.Clear([System.Drawing.Color]::White)
  return @($bmp, $g)
}
$black = [System.Drawing.Color]::Black
function Font($sz, $style = 'Regular') { New-Object System.Drawing.Font('Times New Roman', $sz, [System.Drawing.FontStyle]::$style, [System.Drawing.GraphicsUnit]::Pixel) }
$sf = New-Object System.Drawing.StringFormat; $sf.Alignment = 'Center'; $sf.LineAlignment = 'Center'
$sfL = New-Object System.Drawing.StringFormat; $sfL.Alignment = 'Near'; $sfL.LineAlignment = 'Center'
$sfR = New-Object System.Drawing.StringFormat; $sfR.Alignment = 'Far'; $sfR.LineAlignment = 'Center'

function Box($g, $x, $y, $w, $h, $text, $dashed, $fill, $fs = 30) {
  $pen = New-Object System.Drawing.Pen($black, 3)
  if ($dashed) { $pen.DashStyle = 'Dash' }
  $br = New-Object System.Drawing.SolidBrush($fill)
  $g.FillRectangle($br, $x, $y, $w, $h)
  $g.DrawRectangle($pen, $x, $y, $w, $h)
  $rect = New-Object System.Drawing.RectangleF(($x + 8), $y, ($w - 16), $h)
  $g.DrawString($text, (Font $fs), [System.Drawing.Brushes]::Black, $rect, $sf)
}
function Arrow($g, $x1, $y1, $x2, $y2) {
  $pen = New-Object System.Drawing.Pen($black, 3)
  $pen.CustomEndCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(5, 5)
  $g.DrawLine($pen, $x1, $y1, $x2, $y2)
}
$grey = [System.Drawing.Color]::FromArgb(235, 235, 235)
$white = [System.Drawing.Color]::White

# ---------------- Fig 1: architecture (column width 3.4in x 4.6in @300dpi = 1020 x 1380)
$bmp, $g = New-Canvas 1020 1250
$cx = 510; $bw = 560; $bh = 100; $x = $cx - $bw / 2
$steps = @(
  @('Medical image + question', $false),
  @(('Region localizer' + "`n" + 'detector boxes + MedSAM masks'), $false),
  @('Set-of-Mark overlay  (R1 ... Rk)', $false),
  @(('Dual VLM, JSON contract' + "`n" + 'Qwen2.5-VL-3B / MedGemma-4B, N = 5'), $false),
  @(('Claim verifier' + "`n" + 'drop findings without a valid region'), $false),
  @(('Confidence gate' + "`n" + 'c = w1 a + w2 (1 - Hn) + w3 rho'), $false)
)
$y = 20
for ($i = 0; $i -lt $steps.Count; $i++) {
  Box $g $x $y $bw $bh $steps[$i][0] $false $grey 30
  if ($i -lt $steps.Count - 1) { Arrow $g $cx ($y + $bh) $cx ($y + $bh + 42) }
  $y += $bh + 42
}
# outputs
$yo = $y
Arrow $g ($cx - 120) ($yo - 42) ($cx - 220) ($yo + 10)
Arrow $g ($cx + 120) ($yo - 42) ($cx + 220) ($yo + 10)
Box $g 60 ($yo + 10) 400 90 ('c >= tau :' + "`n" + 'answer + regions') $false $grey 30
Box $g 560 ($yo + 10) 400 90 ('c < tau :' + "`n" + 'abstain (uncertain)') $false $grey 30
# not-yet-done items (dashed)
$yd = $yo + 150
Box $g 60 $yd 900 70 'Weights and tau fitted on real SLAKE outputs (not done)' $true $white 28
Box $g 60 ($yd + 85) 900 70 'Integrated pipeline on held-out test cohort (not done)' $true $white 28
$bmp.Save((Join-Path $out 'fig1_architecture.png'), [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $bmp.Dispose()

# ---------------- Fig 2: pilot accuracy bars (1020 x 640)
$bmp, $g = New-Canvas 1020 660
$l = 130; $r = 990; $t = 40; $b = 520
$axis = New-Object System.Drawing.Pen($black, 3)
$g.DrawLine($axis, $l, $t, $l, $b); $g.DrawLine($axis, $l, $b, $r, $b)
$grid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(200, 200, 200), 1)
foreach ($v in 0, 0.2, 0.4, 0.6, 0.8, 1.0) {
  $yy = $b - $v * ($b - $t)
  if ($v -gt 0) { $g.DrawLine($grid, $l, $yy, $r, $yy) }
  $g.DrawString(('{0:P0}' -f $v), (Font 26), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(0, ($yy - 20), ($l - 12), 40)), $sfR)
}
$groups = @('Overall', 'Open', 'Closed')
$q = @(0.6, 0.4, 0.8); $m = @(0.7, 0.4, 1.0)
$qt = @('6/10', '2/5', '4/5'); $mt = @('7/10', '2/5', '5/5')
$gw = ($r - $l) / 3; $barw = 120
$cq = [System.Drawing.Color]::FromArgb(160, 160, 160); $cm = [System.Drawing.Color]::FromArgb(50, 50, 50)
for ($i = 0; $i -lt 3; $i++) {
  $gx = $l + $i * $gw + $gw / 2
  foreach ($k in 0, 1) {
    if ($k -eq 0) { $val = $q[$i]; $col = $cq; $lab = $qt[$i]; $bx = $gx - $barw - 8 } else { $val = $m[$i]; $col = $cm; $lab = $mt[$i]; $bx = $gx + 8 }
    $h = $val * ($b - $t)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($col)), $bx, ($b - $h), $barw, $h)
    $g.DrawRectangle($axis, $bx, ($b - $h), $barw, $h)
    $g.DrawString($lab, (Font 26), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(($bx - 20), ($b - $h - 44), ($barw + 40), 40)), $sf)
  }
  $g.DrawString($groups[$i], (Font 28), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(($gx - 150), ($b + 8), 300, 44)), $sf)
}
# legend
$g.FillRectangle((New-Object System.Drawing.SolidBrush($cq)), 250, 590, 36, 28); $g.DrawRectangle($axis, 250, 590, 36, 28)
$g.DrawString('Qwen2.5-VL-3B', (Font 28), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(296, 584, 300, 40)), $sfL)
$g.FillRectangle((New-Object System.Drawing.SolidBrush($cm)), 590, 590, 36, 28); $g.DrawRectangle($axis, 590, 590, 36, 28)
$g.DrawString('MedGemma-4B', (Font 28), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(636, 584, 300, 40)), $sfL)
$bmp.Save((Join-Path $out 'fig2_pilot_accuracy.png'), [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $bmp.Dispose()

# ---------------- Fig 3: cohort flow (1020 x 1000)
$bmp, $g = New-Canvas 1020 1040
$g.DrawString('Test (evaluation)', (Font 30 'Bold'), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(0, 5, 1020, 40)), $sf)
Box $g 160 55 700 80 'English test questions: 1,061' $false $grey 30
Arrow $g 510 135 510 175
Box $g 160 175 700 100 ('After removing the 10 pilot questions' + "`n" + 'and their 9 images: 945 (87 images)') $false $grey 30
Arrow $g 400 275 270 320; Arrow $g 620 275 750 320
Box $g 40 320 440 90 ('With GT regions: 864' + "`n" + '(Track B eligible)') $false $grey 28
Box $g 540 320 440 90 ('No GT regions: 81' + "`n" + '(answer accuracy only)') $false $grey 28
$g.DrawString('Calibration (validation split)', (Font 30 'Bold'), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(0, 470, 1020, 40)), $sf)
Box $g 160 520 700 80 'English validation questions: 1,053' $false $grey 30
Arrow $g 510 600 510 640
Box $g 160 640 700 100 ('After removing 15 pilot images' + "`n" + '(162 questions): 891') $false $grey 30
Arrow $g 400 740 270 785; Arrow $g 620 740 750 785
Box $g 40 785 440 100 ('Fit: 719 questions' + "`n" + '65 images') $false $grey 28
Box $g 540 785 440 100 ('Assessment: 172 questions' + "`n" + '16 images') $false $grey 28
$g.DrawString('Fit and assessment share no image; the test split is never used.', (Font 24 'Italic'), [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(0, 930, 1020, 40)), $sf)
$bmp.Save((Join-Path $out 'fig3_cohorts.png'), [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $bmp.Dispose()
Get-ChildItem $out | Select Name, Length
