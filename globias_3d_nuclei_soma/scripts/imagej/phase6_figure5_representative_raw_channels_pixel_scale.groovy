import ij.IJ
import ij.ImagePlus
import ij.plugin.Duplicator
import ij.io.FileSaver
import ij.process.ImageProcessor
import ij.process.ColorProcessor
import java.awt.Color
import java.awt.Font
import java.io.File

// -------------------------
// Hardcoded input/output paths
// -------------------------
String dapiPath = '/data/tasks/Globias_workshop/DAPI/F0003_DAPI.tif'
String somaPath = '/data/tasks/Globias_workshop/stain/F0003_stain.tif'
String outputPng = '/app/data/projects/globias_3d_nuclei_soma/figures/figure5_representative_raw_channels_pixel_scale.png'
String outputTxt = '/app/data/projects/globias_3d_nuclei_soma/figures/figure5_representative_raw_channels_pixel_scale_caption.txt'

// -------------------------
// Utility: percentile from histogram
// -------------------------
double percentileFromHistogram(int[] hist, double histMin, double histMax, double percentile) {
    long total = 0
    for (int h : hist) total += h
    if (total <= 0) return histMin

    long target = Math.round((percentile / 100.0) * (total - 1))
    long cumulative = 0
    int bin = 0
    for (int i = 0; i < hist.length; i++) {
        cumulative += hist[i]
        if (cumulative > target) {
            bin = i
            break
        }
    }
    if (hist.length <= 1) return histMin
    return histMin + (bin as double) * (histMax - histMin) / (hist.length - 1)
}

// -------------------------
// Utility: create display-only RGB panel from one z slice
// -------------------------
ImageProcessor makeDisplayPanel(ImagePlus stackImp, int zPlane1Based, String labelText) {
    ImagePlus zImp = new Duplicator().run(stackImp, 1, 1, zPlane1Based, zPlane1Based, 1, 1)
    if (zImp == null) {
        throw new IllegalStateException('VERIFICATION FAILED: could not duplicate requested Z plane for ' + labelText)
    }

    ImageProcessor ip = zImp.getProcessor().duplicate()
    def st = ip.getStatistics()
    int[] hist = st.histogram

    double pLow = percentileFromHistogram(hist, st.histMin, st.histMax, 1.0)
    double pHigh = percentileFromHistogram(hist, st.histMin, st.histMax, 99.8)
    if (!(pHigh > pLow)) {
        pLow = st.min
        pHigh = st.max
    }
    if (!(pHigh > pLow)) {
        pLow = 0
        pHigh = 65535
    }

    ip.setMinAndMax(pLow, pHigh)
    ImageProcessor ip8 = ip.convertToByte(true)
    ImageProcessor rgb = ip8.convertToRGB()

    int w = rgb.getWidth()
    int h = rgb.getHeight()

    // Channel label
    rgb.setColor(Color.white)
    rgb.setFont(new Font('SansSerif', Font.BOLD, 28))
    rgb.drawString(labelText, 24, 42)

    // 100 px scale bar and exact label
    int barLen = 100
    int barH = 6
    int pad = 28
    int barX = w - barLen - pad
    int barY = h - pad
    rgb.fillRect(barX, barY - barH, barLen, barH)
    rgb.setFont(new Font('SansSerif', Font.PLAIN, 24))
    rgb.drawString('100 px', barX + 8, barY - 12)

    zImp.changes = false
    zImp.close()

    return rgb
}

// -------------------------
// Main
// -------------------------
File dapiFile = new File(dapiPath)
File somaFile = new File(somaPath)
if (!dapiFile.exists()) throw new IllegalArgumentException('Missing input file: ' + dapiPath)
if (!somaFile.exists()) throw new IllegalArgumentException('Missing input file: ' + somaPath)

ImagePlus dapiRaw = IJ.openImage(dapiPath)
ImagePlus somaRaw = IJ.openImage(somaPath)
if (dapiRaw == null) throw new IllegalStateException('VERIFICATION FAILED: failed to open DAPI stack: ' + dapiPath)
if (somaRaw == null) throw new IllegalStateException('VERIFICATION FAILED: failed to open soma stack: ' + somaPath)

// Work only on duplicates
ImagePlus imp2 = dapiRaw.duplicate()
ImagePlus somaDup = somaRaw.duplicate()
imp2.setTitle('F0003 DAPI duplicate')
somaDup.setTitle('F0003 Soma duplicate')

if (imp2.getWidth() != somaDup.getWidth() || imp2.getHeight() != somaDup.getHeight()) {
    throw new IllegalStateException('VERIFICATION FAILED: DAPI and soma dimensions differ; identical panel extent cannot be guaranteed')
}
if (imp2.getNSlices() != somaDup.getNSlices()) {
    throw new IllegalStateException('VERIFICATION FAILED: DAPI and soma Z-slice counts differ; representative paired plane cannot be guaranteed')
}

int nSlices = imp2.getNSlices()
if (nSlices < 1) throw new IllegalStateException('VERIFICATION FAILED: input stack has no slices')

// Representative central interior plane (1-based)
int zPlane = (int)Math.round(nSlices / 2.0)
if (zPlane < 1) zPlane = 1
if (zPlane > nSlices) zPlane = nSlices

ImageProcessor dapiPanel = makeDisplayPanel(imp2, zPlane, 'DAPI (raw)')
ImageProcessor somaPanel = makeDisplayPanel(somaDup, zPlane, 'Soma marker (raw)')

int w = dapiPanel.getWidth()
int h = dapiPanel.getHeight()
if (somaPanel.getWidth() != w || somaPanel.getHeight() != h) {
    throw new IllegalStateException('VERIFICATION FAILED: generated panel dimensions are inconsistent')
}

int margin = 20
int gap = 20
int captionH = 72
int canvasW = margin + w + gap + w + margin
int canvasH = margin + h + 16 + captionH

ColorProcessor canvas = new ColorProcessor(canvasW, canvasH)
canvas.setColor(Color.black)
canvas.fill()

int leftX = margin
int rightX = margin + w + gap
int y = margin
canvas.insert(dapiPanel, leftX, y)
canvas.insert(somaPanel, rightX, y)

canvas.setColor(Color.white)
canvas.setFont(new Font('SansSerif', Font.PLAIN, 24))
String note = 'Note: physical calibration uncertain; scale bars are pixel-based only (100 px).'
canvas.drawString(note, margin, margin + h + 40)

ImagePlus figureImp = new ImagePlus('Figure5_representative_raw_channels', canvas)

File outDir = new File(new File(outputPng).getParent())
outDir.mkdirs()

boolean pngSaved = new FileSaver(figureImp).saveAsPng(outputPng)
if (!pngSaved) throw new IllegalStateException('VERIFICATION FAILED: PNG save returned false: ' + outputPng)

File pngFile = new File(outputPng)
if (!pngFile.exists() || pngFile.length() <= 0) {
    throw new IllegalStateException('VERIFICATION FAILED: output PNG was not created or is empty: ' + outputPng)
}

String displayAdjustment = 'Per-panel robust display contrast from selected Z plane histogram: p1.0 to p99.8, fallback to min/max if needed; display-only on duplicate slices.'
String captionText = ''
captionText += 'Figure: Representative raw channel panel for F0003\n'
captionText += 'Output PNG: ' + outputPng + '\n'
captionText += 'Source DAPI stack: ' + dapiPath + '\n'
captionText += 'Source soma stack: ' + somaPath + '\n'
captionText += 'Selected representative Z plane (1-based): ' + zPlane + ' of ' + nSlices + '\n'
captionText += 'Panel extent/crop: full frame, identical for both channels (' + w + 'x' + h + ' px)\n'
captionText += 'Display adjustment: ' + displayAdjustment + '\n'
captionText += 'Scale bar: length=100 pixels, thickness=6 px, white, lower-right on each panel, label text exactly "100 px"\n'
captionText += 'Calibration caveat: Physical calibration is uncertain; bars are shown in pixels only and not converted to microns.\n'

File txtFile = new File(outputTxt)
txtFile.text = captionText
if (!txtFile.exists() || txtFile.length() <= 0) {
    throw new IllegalStateException('VERIFICATION FAILED: metadata/caption file missing or empty: ' + outputTxt)
}

// Close temporary images without saving changes
dapiRaw.changes = false
somaRaw.changes = false
somaDup.changes = false

dapiRaw.close()
somaRaw.close()
somaDup.close()

// Show duplicate and final panel for supervisor inspection
imp2.show()
figureImp.show()

println 'SUCCESS: Saved representative raw-channel panel PNG to ' + outputPng
println 'SUCCESS: Saved caption/metadata text to ' + outputTxt
