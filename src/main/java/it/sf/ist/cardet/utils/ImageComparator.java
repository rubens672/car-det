package it.sf.ist.cardet.utils;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.opencv_core.DMatchVector;
import org.bytedeco.javacpp.opencv_core.KeyPointVector;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_features2d.BFMatcher;
import org.bytedeco.javacpp.opencv_xfeatures2d.SURF;
import org.bytedeco.javacpp.opencv_core.DMatch;

import static org.bytedeco.javacpp.opencv_imgproc.HISTCMP_INTERSECT;
import static org.bytedeco.javacpp.opencv_imgproc.CV_TM_SQDIFF;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import static org.bytedeco.javacpp.opencv_imgproc.compareHist;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;
import static org.bytedeco.javacpp.opencv_core.minMaxLoc;
import static org.bytedeco.javacpp.opencv_core.NORM_L2;
import static org.bytedeco.javacpp.opencv_highgui.imshow;

/**
 * Retrieving similar images using histogram comparison.
 * 
 * @author Antonio Berti
 * 
 * @version 1.0
 *
 */
public class ImageComparator {

	private float _minRange  = 0.0f;
	private float _maxRange  = 255.0f;
	private int numberOfBins = 256;

	/**
	 * Compare the reference image with the given input image and return similarity score.
	 * 
	 * Compute histogram match and normalize by image size. 
	 * 1 means perfect match.
	 */
	public double comparingHistograms(Mat referenceImage, Mat image){
		Mat referenceHistogram = getHistogram(referenceImage);
		Mat inputH = getHistogram(image);
		int imageSize = image.cols() * image.rows();
		double similarityScore = compareHist(referenceHistogram, inputH, HISTCMP_INTERSECT);
		double score = similarityScore / imageSize;
		System.out.println(String.format("%6.4f", score));
//		imshow(String.format("%6.4f", score),inputH);
		return score;
	}
	
	public String templateMatching(Mat referenceImage, Mat image) {
		// perform template matching
		Mat result = new Mat();
		matchTemplate(referenceImage, image, result, CV_TM_SQDIFF);
		
		  // find most similar location
		DoublePointer minVal = new DoublePointer(1);
		DoublePointer maxVal = new DoublePointer(1);
		Point minPt  = new Point();
		Point maxPt  = new Point();
		minMaxLoc(result, minVal, maxVal, minPt, maxPt, null);
		
		int ix = minPt.x();
		int iy = minPt.y();
		int ax = maxPt.x();
		int ay = maxPt.y();
		
		return result.toString();
	}
	
	public void bruteForceMatcher(Mat queryDescriptors, Mat trainDescriptors){
		 // Construction of the SURF feature detector
		SURF surf = SURF.create(3000, 4, 2, false, false);
		
		 // Detection of the SURF features
		KeyPointVector keypointsRight = new KeyPointVector();
		KeyPointVector keypointsLeft  = new KeyPointVector();
		surf.detect(queryDescriptors, keypointsRight);
		surf.detect(trainDescriptors, keypointsLeft);
		  
		// Extraction of the SURF descriptors
		Mat descriptorsRight = new Mat();
		Mat descriptorsLeft  = new Mat();
		surf.compute(queryDescriptors, keypointsRight, descriptorsRight);
		surf.compute(trainDescriptors, keypointsLeft, descriptorsLeft);
		
		
		DMatchVector matches = new DMatchVector();
		// Create feature matcher
		BFMatcher matcher = new BFMatcher(NORM_L2, false);
		matcher.match(descriptorsRight, descriptorsLeft, matches);
		System.out.println("DMatchVector | size:" + matches.size() + " - capacity: " + matches.capacity());

	}

	/**
	 * Computes histogram of an image.
	 *
	 * @param image input image
	 * @return OpenCV histogram object
	 */
	public Mat getHistogram(Mat image){

		// Compute histogram
		Mat hist = new Mat();

		// Since C++ `calcHist` is using arrays of arrays we need wrap to do some wrapping
		// in `IntPointer` and `PointerPointer` objects.
		IntPointer intPtrChannels = new IntPointer(0, 1, 2);
		IntPointer intPtrHistSize = new IntPointer(numberOfBins, numberOfBins, numberOfBins);
		float[] histRange = new float[]{_minRange, _maxRange};
		PointerPointer<FloatPointer> ptrPtrHistRange = new PointerPointer<FloatPointer>(histRange, histRange, histRange);
		calcHist(image,
				1, // histogram of 1 image only
				intPtrChannels, // the channel used
				new Mat(), // no mask is used
				hist, // the resulting histogram
				3, // it is a 3D histogram
				intPtrHistSize, // number of bins
				ptrPtrHistRange, // pixel value range
				true, // uniform
				false); // no accumulation

		return hist;
	}
}
