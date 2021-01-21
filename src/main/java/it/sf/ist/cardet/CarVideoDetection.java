package it.sf.ist.cardet;

import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_core.Size;

/**------------
 * - Bytedeco -
 * ------------
 * - Bringing compute-intensive science, multimedia, computer vision, deep learning, etc to the Java platform
 * https://github.com/bytedeco
 * 
 * - JavaCV uses wrappers from the JavaCPP Presets of commonly used libraries by researchers in the field of computer vision 
 * https://github.com/bytedeco/javacv
 * 
 * - esempi javacv
 * https://github.com/bytedeco/javacv/tree/master/samples
 * 1661
 * 
 * @author Antonio Berti
 * 
 * @version 1.0
 *
 */
public class CarVideoDetection {

	private static final String AUTONOMOUS_DRIVING_RAMOK_TECH = "ist AI Car Detection";
	private volatile Frame[] videoFrame = new Frame[1];
	private volatile Mat[] v = new Mat[1];
	private Thread thread;
	private volatile boolean stop = false;
	private String winname;

	private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

	public static void main(String[] args) throws java.lang.Exception {
//				new CarVideoDetection().startRealTimeVideoDetection("resources/videoSample.mp4");
		new CarVideoDetection().startImageRealTimeVideo();
		//		new CarVideoDetection().startConvertToMp4();
	}

	public void startRealTimeVideoDetection(String videoFileName) throws java.lang.Exception {

		File f = new File(videoFileName);

		FFmpegFrameGrabber grabber;
		grabber = new FFmpegFrameGrabber(f);
		grabber.start();
		
		while (!stop) {
			videoFrame[0] = grabber.grab();
			if (videoFrame[0] == null) {
				stop();
				break;
			}
			v[0] = new OpenCVFrameConverter.ToMat().convert(videoFrame[0]);
			if (v[0] == null) {
				continue;
			}
			if (winname == null) {
				winname = AUTONOMOUS_DRIVING_RAMOK_TECH + ThreadLocalRandom.current().nextInt();
			}

//			resize(v[0], v[0], new Size(416, 416));
			
//			if (thread == null) {
//				thread = new Thread(() -> {
//					while (videoFrame[0] != null && !stop) {
//						try {
//							//							TinyYoloPrediction.getINSTANCE().markWithBoundingBox(v[0], videoFrame[0].imageWidth, videoFrame[0].imageHeight, true, winname);
//							TinyYoloPrediction.getINSTANCE().markWithBoundingBox(v[0], 416, 416, true, winname, grabber.getFrameNumber());
//						} catch (java.lang.Exception e) {
//							throw new RuntimeException(e);
//						}
//					}
//				});
//				thread.start();
//			}

			TinyYoloPrediction.getINSTANCE().markWithBoundingBox(v[0], videoFrame[0].imageWidth, videoFrame[0].imageHeight, true, winname, grabber.getFrameNumber());
//			TinyYoloPrediction.getINSTANCE().markWithBoundingBox(v[0], 416, 416, true, winname, grabber.getFrameNumber());

			
//			resize(v[0], v[0], new Size(640, 480));
			imshow(winname, v[0]);

			char key = (char) waitKey(10);
			// Exit this loop on escape:
			if (key == 27) {
				stop();
				break;
			}
		}
	}

	public void stop() {
		if (!stop) {
			stop = true;
			destroyAllWindows();
		}
	}

	public void stringToDom(String img, String dir, String fileName) 
			throws IOException {
		java.io.FileWriter fw = new java.io.FileWriter(dir + "/" + fileName + ".xml");
		fw.write(xml.replaceAll("FILE_NAME", fileName + ".jpg").replace("FILE_DIR", img));
		fw.close();
	}

	public void startImageRealTimeVideo() throws java.lang.Exception {
		System.out.println("startImageRealTimeVideo");

		String videoFileName = "422_P20I_1012_16N_0030_0100";
		String resources = "resources/";
		String videoFileNamePath = "resources/" + videoFileName + ".avi";

		String images = "images_III_" + videoFileName + "/images";
		String annotations = "images_III_" + videoFileName + "/Annotations";

		File imagesDir = new File("C:/" + images);
		if(!imagesDir.exists()) {
			System.out.println("create dir " + videoFileName + images);
			imagesDir.mkdirs();
		}

		File annotationsDir = new File("C:/" + annotations);
		if(!annotationsDir.exists()) {
			System.out.println("create dir " + videoFileName + annotations);
			annotationsDir.mkdirs();
		}


		CanvasFrame mainframe = new CanvasFrame("Car Video Detection - ist", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setCanvasSize(416, 416);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);


		File f = new File(videoFileNamePath);

		FFmpegFrameGrabber grabber;
		grabber = new FFmpegFrameGrabber(f);
		grabber.setMaxDelay(10);
		grabber.start();
		
		//ultimo frame 29980
		int frame = 20000;

		for(int i=0; i<500;i++, frame+=20) {
			//		while (!stop) {int i = 0;

			grabber.setFrameNumber(frame);

			videoFrame[0] = grabber.grab();
			if (videoFrame[0] == null) {
				stop();
				break;
			}
			Mat mat = new OpenCVFrameConverter.ToMat().convert(videoFrame[0]);
			if (mat == null) {
				continue;
			}

			resize(mat, mat, new Size(416, 416));
			
			String fileName = "img_"+ videoFileName + "_" + frame + ".jpg"; 
			imwrite("C:/" + images + "/" + fileName, mat);
			System.out.println(i + " " + fileName);

			stringToDom(imagesDir.getAbsolutePath(), annotationsDir.getAbsolutePath(), "img_"+ videoFileName + "_" + frame );

			mainframe.showImage(converter.convert(mat));

			Thread.sleep(150);

			//			System.out.print("getVideoStream " + grabber.getVideoStream());
			//			System.out.print(" - getSampleRate " + grabber.getSampleRate());
			//			System.out.print(" - getFrameRate " + grabber.getFrameRate());
			//			System.out.print(" - getFrameNumber " + grabber.getFrameNumber());
			//			System.out.print(" - getTimestamp " + grabber.getTimestamp());
			//			System.out.println();

			//			System.out.println(fileName);
		}

		grabber.stop();
		grabber.release();

		System.out.println("wait...");
		Thread.sleep(2000);
		System.out.println("done...");
		System.exit(0);
	}

	public void startConvertToMp4() throws java.lang.Exception {

		CanvasFrame mainframe = new CanvasFrame("Convert To Mp4", CanvasFrame.getDefaultGamma() / 2.2);
		mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		mainframe.setCanvasSize(600, 600);
		mainframe.setLocationRelativeTo(null);
		mainframe.setVisible(true);

		FFmpegFrameGrabber grabber = new FFmpegFrameGrabber("E:\\DCIM\\100CASIO\\CIMG0791.AVI");
		grabber.start();



		FrameRecorder recorder = new FFmpegFrameRecorder("E:\\DCIM\\100CASIO\\1_CIMG0791.mp4", grabber.getImageWidth(), grabber.getImageHeight());
		recorder.setSampleRate(grabber.getSampleRate());
		//	    recorder.setPixelFormat(grabber.getPixelFormat());
		recorder.start();

		Frame frame;
		while ((frame = grabber.grabFrame()) != null) {
			Mat mat = new OpenCVFrameConverter.ToMat().convert(frame);
			if (mat == null) {
				continue;
			}
			mainframe.showImage(converter.convert(mat));

			recorder.record(frame);
		}
		recorder.stop();
		grabber.stop();

		grabber.close();
		recorder.close();

		System.exit(0);
	}

	private String xml = 
			"<annotation>\n"+
					"	<folder>images</folder>\n"+
					"	<filename>FILE_NAME</filename>\n"+
					"	<path>FILE_DIR\\FILE_NAME</path>\n"+
					"	<source>\n"+
					"		<database>Unknown</database>\n"+
					"	</source>\n"+
					"	<size>\n"+
					"		<width>640</width>\n"+
					"		<height>480</height>\n"+
					"		<depth>3</depth>\n"+
					"	</size>\n"+
					"	<segmented>0</segmented>\n"+
					"	<object>\n"+
					"		<name>car</name>\n"+
					"		<pose>Unspecified</pose>\n"+
					"		<truncated>0</truncated>\n"+
					"		<difficult>0</difficult>\n"+
					"		<bndbox>\n"+
					"			<xmin>80</xmin>\n"+
					"			<ymin>43</ymin>\n"+
					"			<xmax>107</xmax>\n"+
					"			<ymax>69</ymax>\n"+
					"		</bndbox>\n"+
					"	</object>\n"+
					"	<object>\n"+
					"		<name>car</name>\n"+
					"		<pose>Unspecified</pose>\n"+
					"		<truncated>0</truncated>\n"+
					"		<difficult>0</difficult>\n"+
					"		<bndbox>\n"+
					"			<xmin>72</xmin>\n"+
					"			<ymin>94</ymin>\n"+
					"			<xmax>107</xmax>\n"+
					"			<ymax>128</ymax>\n"+
					"		</bndbox>\n"+
					"	</object>\n"+
					"	<object>\n"+
					"		<name>car</name>\n"+
					"		<pose>Unspecified</pose>\n"+
					"		<truncated>0</truncated>\n"+
					"		<difficult>0</difficult>\n"+
					"		<bndbox>\n"+
					"			<xmin>55</xmin>\n"+
					"			<ymin>156</ymin>\n"+
					"			<xmax>109</xmax>\n"+
					"			<ymax>207</ymax>\n"+
					"		</bndbox>\n"+
					"	</object>\n"+
					"	<object>\n"+
					"		<name>car</name>\n"+
					"		<pose>Unspecified</pose>\n"+
					"		<truncated>0</truncated>\n"+
					"		<difficult>0</difficult>\n"+
					"		<bndbox>\n"+
					"			<xmin>34</xmin>\n"+
					"			<ymin>237</ymin>\n"+
					"			<xmax>111</xmax>\n"+
					"			<ymax>314</ymax>\n"+
					"		</bndbox>\n"+
					"	</object>\n"+
					"	<object>\n"+
					"		<name>car</name>\n"+
					"		<pose>Unspecified</pose>\n"+
					"		<truncated>0</truncated>\n"+
					"		<difficult>0</difficult>\n"+
					"		<bndbox>\n"+
					"			<xmin>12</xmin>\n"+
					"			<ymin>340</ymin>\n"+
					"			<xmax>112</xmax>\n"+
					"			<ymax>439</ymax>\n"+
					"		</bndbox>\n"+
					"	</object>\n"+
					"</annotation>\n";
}