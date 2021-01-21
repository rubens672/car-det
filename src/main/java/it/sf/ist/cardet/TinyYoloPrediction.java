package it.sf.ist.cardet;

import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import it.sf.ist.cardet.utils.ImageComparator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.line;
import static org.bytedeco.javacpp.opencv_imgproc.circle;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;
import static org.bytedeco.javacpp.opencv_imgproc.FONT_HERSHEY_DUPLEX;


//import org.nd4j.jita.conf.CudaEnvironment;

/*
 * 9h5p7y8TnYcNmNq
 */

public class TinyYoloPrediction {

	private ComputationGraph preTrained;
	private List<DetectedObject> predictedObjects;
	private HashMap<Integer, String> map;

		static {
			CudaEnvironment.getInstance().getConfiguration()
			.allowMultiGPU(true)// key option enabled
			.setMaximumDeviceCache(6L * 1024L * 1024L * 1024L)// we're allowing larger memory caches
			.allowCrossDeviceAccess(true);// cross-device access is used for faster model averaging over pcie
		}

	private TinyYoloPrediction() {
		try {
			//preTrained = (ComputationGraph)YOLO2.builder().build().initPretrained();
//			preTrained = ModelSerializer.restoreComputationGraph("model/best_pre_TinyYOLOcardet_mille_fotogrammi_3.bin");
			preTrained = ModelSerializer.restoreComputationGraph("model/YOLO2_images_III_422_P20I_1012_16N_0030_0100.bin");
//			preTrained = ModelSerializer.restoreComputationGraph("model/TinyYOLOcardet_gpu.bin");
			//			preTrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
			prepareLabels();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static final TinyYoloPrediction INSTANCE = new TinyYoloPrediction();

	public static TinyYoloPrediction getINSTANCE() {
		return INSTANCE;
	}

	public void markWithBoundingBox(Mat file, int imageWidth, int imageHeight, boolean newBoundingBOx,String winName, int frame) throws Exception {
		int width = 416;
		int height = 416;
		int gridWidth = 13;
		int gridHeight = 13;
		double detectionThreshold = 0.2;
		
//		<xmin>232</xmin>
//		<ymin>111</ymin>
//		<xmax>259</xmax>
//		<ymax>137</ymax>
//		        rectangle(file, new Point(232, 111),   new Point(259, 137), Scalar.WHITE, CV_FILLED , 8, 0);
		//		rectangle(file, new Point(0, 240), new Point(416, 416), Scalar.WHITE, CV_FILLED , 8, 0);

		Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) preTrained.getOutputLayer(0);
		if (newBoundingBOx) {
			try {
				INDArray indArray = prepareImage(file, width, height);
				INDArray results = preTrained.outputSingle(indArray);
				predictedObjects = outputLayer.getPredictedObjects(results, detectionThreshold);
//				                System.out.println("results = " + predictedObjects);
				markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight, frame);
			}catch(Exception e) {
				e.printStackTrace();
				throw new Exception(e);
			}

		} else {
			markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight, frame);
		} 
		
//		circle(file, new Point(534, 32), 3, Scalar.BLUE, -1, 8, 0); //punto di fuga
//		circle(file, new Point(305, 240), 3, Scalar.BLUE, -1, 8, 0);//(xp,yp)
//		circle(file, new Point(390, 240), 3, Scalar.BLUE, -1, 8, 0);//(x1,y1)
//		circle(file, new Point(470, 240), 3, Scalar.BLUE, -1, 8, 0);//(x1,y1)

//		line(file, new Point(0, 200), new Point(imageWidth, 200), Scalar.CYAN);
//		line(file, new Point(0, 180), new Point(imageWidth, 180), Scalar.CYAN);
//		line(file, new Point(0, 210), new Point(imageWidth, 210), Scalar.YELLOW);
		line(file, new Point(0, 280), new Point(imageWidth, 280), Scalar.YELLOW);

//		circle(file, new Point(150, 230), 3, Scalar.RED, -1, 8, 0);
//		circle(file, new Point(222, 230), 3, Scalar.RED, -1, 8, 0);
//
//		circle(file, new Point(150, 260), 3, Scalar.RED, -1, 8, 0);
//		circle(file, new Point(222, 260), 3, Scalar.RED, -1, 8, 0);

		putText(file, count.toString(), new Point(4 , 40), FONT_HERSHEY_DUPLEX, 1.3, Scalar.GREEN);
		//        imshow(winName, file);
	}
	
	private synchronized boolean checkCount(int Ox, int Oy, int i) {
		boolean ret = false;
		int[] xy = getxy(i);
		if(Oy < 260 && Oy >= 240) { 
			setxy(new int[]{Ox, Oy}, i);
			if(xy == null) { 
				count++; 
				ret = true; 
			}
			
		}else if(Oy < 235 && Oy >= 215) { 
			setxy(null, i); 
		}
		return ret;
	}

	private INDArray prepareImage(Mat file, int width, int height) throws IOException {
		NativeImageLoader loader = new NativeImageLoader(height, width, 3);
		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
		INDArray indArray = loader.asMatrix(file);
		imagePreProcessingScaler.transform(indArray);
		return indArray;
	}

	private void prepareLabels() {
		if (map == null) {
			map = new HashMap<>();
			map.put(0, "car");
			map.put(1, "truck");
		}
	}

	private void prepareLabelsOrig() {
		if (map == null) {
			String s = "aeroplane\n" + "bicycle\n" + "bird\n" + "boat\n" + "bottle\n" + "bus\n" + "car\n" +
					"cat\n" + "chair\n" + "cow\n" + "diningtable\n" + "dog\n" + "horse\n" + "motorbike\n" +
					"person\n" + "pottedplant\n" + "sheep\n" + "sofa\n" + "train\n" + "tvmonitor";
			String istcardetLabels = "car\n" + "truck";
			String[] split = istcardetLabels.split("\\n");
			int i = 0;
			map = new HashMap<>();
			for (String s1 : split) {
				map.put(i++, s1);
			}
		}
	}

	private void markWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, int frame) {

		if (predictedObjects == null) {
			return;
		}
		ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);

//		Mat fileClone = file.clone();
		String boundingBoxes = "";
		
		while (!detectedObjects.isEmpty()) {
			Optional<DetectedObject> max = detectedObjects.stream().max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
			if (max.isPresent()) {
				DetectedObject maxObjectDetect = max.get();
				removeObjectsIntersectingWithMax(detectedObjects, maxObjectDetect);
				detectedObjects.remove(maxObjectDetect);
				boundingBoxes += markWithBoundingBox(file, gridWidth, gridHeight, w, h, maxObjectDetect, frame);
			}
		}
		
//		creaDataSet(fileClone, frame, w, h, boundingBoxes);
	}
	
	private void creaDataSet(Mat file, int frame, int width, int height, String boundingBoxes) {
		String videoFileName = "422_P20I_1012_16N_0030_0100";

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
		
		String fileName = "img_"+ videoFileName + "_" + frame + ".jpg"; 
		String xmlName  = "img_"+ videoFileName + "_" + frame + ".xml"; 
		String path = imagesDir + "\\" + fileName;
		//crea annotazione
		String xml = 
				"<annotation>\n"+
						"	<folder>images</folder>\n"+
						"	<filename>" + fileName + "</filename>\n"+
						"	<path>" + path + "</path>\n"+
						"	<source>\n"+
						"		<database>Unknown</database>\n"+
						"	</source>\n"+
						"	<size>\n"+
						"		<width>" + height + "</width>\n"+
						"		<height>" + height + "</height>\n"+
						"		<depth>3</depth>\n"+
						"	</size>\n"+
						"	<segmented>0</segmented>\n" +
						
							boundingBoxes +
							
						"</annotation>\n";
		
		try {
			java.io.FileWriter fw = new java.io.FileWriter(annotationsDir + "/" + xmlName);
			fw.write(xml);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//crea immagine
		imwrite("C:/" + images + "/" + fileName, file);
		System.out.println(frame + " " + fileName);
	}

	private static void removeObjectsIntersectingWithMax(ArrayList<DetectedObject> detectedObjects, DetectedObject maxObjectDetect) {
		double[] bottomRightXY1 = maxObjectDetect.getBottomRightXY();
		double[] topLeftXY1 = maxObjectDetect.getTopLeftXY();
		List<DetectedObject> removeIntersectingObjects = new ArrayList<>();
		for (DetectedObject detectedObject : detectedObjects) {
			double[] topLeftXY = detectedObject.getTopLeftXY();
			double[] bottomRightXY = detectedObject.getBottomRightXY();
			double iox1 = Math.max(topLeftXY[0], topLeftXY1[0]);
			double ioy1 = Math.max(topLeftXY[1], topLeftXY1[1]);

			double iox2 = Math.min(bottomRightXY[0], bottomRightXY1[0]);
			double ioy2 = Math.min(bottomRightXY[1], bottomRightXY1[1]);

			double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

			double box1_area = (bottomRightXY1[1] - topLeftXY1[1]) * (bottomRightXY1[0] - topLeftXY1[0]);
			double box2_area = (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

			double union_area = box1_area + box2_area - inter_area;
			double iou = inter_area / union_area;


			if (iou > 0.5) {
				removeIntersectingObjects.add(detectedObject);
			}

		}
		detectedObjects.removeAll(removeIntersectingObjects);
	}

	private Integer count = 0;
	
	private int countImage = 0;


	private String markWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, DetectedObject obj, int frame) {
		double[] xy1 = obj.getTopLeftXY();
		double[] xy2 = obj.getBottomRightXY();
		double cx = obj.getCenterX();
		double cy = obj.getCenterY();
		
		int Ox = (int) Math.round(w * cx / gridWidth);
		int Oy = (int) Math.round(h * cy / gridHeight);
		if(Oy < 220 || Oy > 240) return "";
		
		int x1 = (int) Math.round(w * xy1[0] / gridWidth);
		int y1 = (int) Math.round(h * xy1[1] / gridHeight);
		int x2 = (int) Math.round(w * xy2[0] / gridWidth);
		int y2 = (int) Math.round(h * xy2[1] / gridHeight);
		
		
		
//		Executors.newCachedThreadPool().submit(() -> {
//			int corsia = ricavaCorsia(Ox, Oy);
//			String corsiaDir = "corsia_";
//			if(corsia == 0) {
//				corsiaDir += "I/";
//			}else if(corsia == 1) {
//				corsiaDir += "II/";
//			}else {
//				corsiaDir += "III/";
//			}
//			
////			Rect rectCrop = new Rect(x1, y1, x2-x1, y2-y1);
//			Mat image_roi = new Mat(file, new Rect(x1, y1, x2-x1, y2-y1)).clone();
//			countImage++;
//			String img = "img_" + countImage + ".jpg";
//			imwrite("C:/testImage/" +corsiaDir + img, image_roi);
//			System.out.println("create " + img);
//		});
		
		
//		imshow("rectCrop", image_roi);
		
//		Executors.newCachedThreadPool().submit(() -> {
		try {
			if(checkCountIV(new Mat(file, new Rect(x1, y1, x2-x1, y2-y1)), Ox, Oy, frame)) {
				putText(file, count.toString(), new Point(x1+10, y1-30), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
				System.out.println(count.toString());
			}
		}catch(Exception e) {
			e.printStackTrace();
		}

//		});
		
		circle(file, new Point(Ox, Oy), 4, Scalar.RED, -1, 8, 0); //centro box
//		rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.RED); //bounding box
//		putText(file, count.toString(), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
		
//		String boundingBox = 
//		"	<object>\n"+
//		"		<name>" + map.get(predictedClass) + "</name>\n"+
//		"		<pose>Unspecified</pose>\n"+
//		"		<truncated>0</truncated>\n"+
//		"		<difficult>0</difficult>\n"+
//		"		<bndbox>\n"+
//		"			<xmin>" + x1 + "</xmin>\n"+
//		"			<ymin>" + y1 + "</ymin>\n"+
//		"			<xmax>" + x2 + "</xmax>\n"+
//		"			<ymax>" + y2 + "</ymax>\n"+
//		"		</bndbox>\n"+
//		"	</object>\n";
//		return boundingBox;
		return "";
	}

	private static HashMap<Integer,int[]> m = new HashMap<Integer,int[]>();
	
	private int[] xy_I;
	private int[] xy_II;
	private int[] xy_III;
	
	private int[] getxy(int i) {
		if(i == 1) return xy_I;
		else if(i == 2) return xy_II;
		else return xy_III;
	}
	
	private void setxy(int[] xy, int i) {
		if(i == 1) xy_I = xy;
		else if(i == 2) xy_II = xy;
		else xy_III = xy;
	}
	
	private Mat[] autoInCorsia = new Mat[4];
	private int[] frameArr = new int[]{0,0,0,0};
	
	private int ricavaCorsia(int Ox, int Oy) {
		//calcolo del coefficente angolare
		//m = (y2-y1) / (x2-x1)
		double x1,x2,y1,y2,m,yp;
		x1 = 349; // punto di fuga
		y1 = 22;  //
		x2 = Ox;
		y2 = Oy;
		m = (y2-y1) / (x2-x1);

		//equazione della retta passante per un punto
		//y - yp = m*(x-yp)
		//da cui si ricava l'equazione per calcolare xp
		//xp = ((y-yp)-(m*x)) / m
		yp = 260; //linea di riferimento 
		int xp = Math.abs((int) Math.round( ((Oy-yp) - (m*Ox)) / m ));

		if(xp <= 150) return 0;
		else if(xp > 222) return 2;
		else return 1;
	}
	
	private synchronized boolean checkCountIV(Mat referenceImage, int Ox, int Oy, int frame) {
		
		//calcolo del coefficente angolare
		//m = (y2-y1) / (x2-x1)
		double x1,x2,y1,y2,m,yp;
		x1 = 349; // punto di fuga, riferimento di tutte le traiettorie
		y1 = 22;  //
		x2 = Ox;
		y2 = Oy;
		m = (y2-y1) / (x2-x1);

		//equazione della retta passante per un punto
		//y - yp = m*(x-xp)
		//da cui si ricava l'equazione per calcolare xp
		//xp = ((y-yp)-(m*x)) / m
		yp = 240; //linea di riferimento
		int xp = Math.abs((int) Math.round( ((Oy-yp) - (m*Ox)) / m ));

		int corsia = -1;
		
		if(xp <= 305) corsia = 0;
		else if(xp > 305 && xp <= 390) corsia = 1;
		else if(xp > 390 && xp <= 470) corsia = 2;
		else return false;//corsia = 3;
		
		if(frame - frameArr[corsia] > 3) { 
			autoInCorsia[corsia] = null; 
		}
		
		Mat image = autoInCorsia[corsia];
		
		if(image == null && Oy < yp && Oy >= 220) {
			autoInCorsia[corsia] = referenceImage;
			frameArr[corsia] = frame;
			count++; 
			return true;
		}else if(image != null && Oy < yp && Oy >= 220) {
			// Compute histogram match and normalize by image size.
		    // 1 means perfect match.
			double score = comparator.comparingHistograms(referenceImage, image);
//			imshow("referenceImage", referenceImage);
//			Oy < 260 && Oy >= 240
			autoInCorsia[corsia] = referenceImage;
			frameArr[corsia] = frame;
			if(score > 0.1) {
				return false;
			}else {
				count++;
				return true;
			}
		}else {
			return false;
		}
//		else if(Oy < 200 && Oy > 180) { 
//			autoInCorsia[corsia] = null;
//		} 
	}
	
	private ImageComparator comparator = new ImageComparator();
	
	private synchronized boolean checkCountIII(int Ox, int Oy) {
		boolean ret = false;
		
		//calcolo del coefficente angolare
		//m = (y2-y1) / (x2-x1)
		double x1,x2,y1,y2,m,yp;
		x1 = 349; // punto di fuga
		y1 = 22;  //
		x2 = Ox;
		y2 = Oy;
		m = (y2-y1) / (x2-x1);
		
		//equazione della retta passante per un punto
		//y - yp = m*(x-yp)
		//da cui si ricava l'equazione per calcolare xp
		//xp = ((y-yp)-(m*x)) / m
		yp = 260; //linea di riferimento
		int xp = (int) Math.round( ((Oy-yp) - (m*Ox)) / m );
		
		if(xp <= 150) ret = checkCount(Ox, Oy, 1);
		else if(xp > 222) ret = checkCount(Ox, Oy, 3);
		else ret = checkCount(Ox, Oy, 2);
		return ret;
	}

	private synchronized boolean checkCountII(int Ox, int Oy) {
		boolean ret = false;
		if(Ox <= 181) ret = checkCount(Ox, Oy, 1);
		else if(Ox > 227) ret = checkCount(Ox, Oy, 3);
		else ret = checkCount(Ox, Oy, 2);
		return ret;
	}


	
	

	private synchronized boolean checkCount(int Ox, int Oy) {
		int size = m.keySet().size();
		//    	  181, 230   243, 230
		//    	
		//    	156, 260    227, 260
		//    	
		//elimina le macchine gi� conteggiate dalla lista
		boolean isOld = false;
		int it = -1;
		for(int i: m.keySet()) {
			int[] xy = m.get(i);
			if(xy[1] < 220) {
				it = i;
				isOld = true;
				break;
			}
		}

		m.remove(it);

		int r = 0;
		it = -1;
		boolean isNew = true;
		for(int i: m.keySet()) {
			int[] xy = m.get(i);
			//distanza fra due punti
			r = (int) Math.round(Math.sqrt( Math.pow(Ox-xy[0], 2) +  Math.pow(Oy-xy[1], 2)));
			if(r<30) {
				it = i;
				isNew = false;
				break;
			}
		}

		if(isNew && Oy < 260 && Oy > 230) {
			count++;
			m.put(count, new int[] {Ox, Oy});
			return true;
		}else {
			m.remove(it);
			m.put(it, new int[] {Ox, Oy});
			return false;
		}
	}

	private String istcardetLabels = "car\n" + "truck";

}