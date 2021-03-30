package tflite;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class ImageSegmenter {
    public static final String TAG = "ImageSegmenter";
    public static  final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private int[] intValues;
    private int[] outFrame;
    private Interpreter.Options tfliteOptions;
    private MappedByteBuffer tfliteModel;
    private  Model model;
    private Interpreter tflite;
    private ByteBuffer segmentedImage;
    private ByteBuffer imgData;
    private Activity activity;

    public ImageSegmenter(Activity activity) throws IOException {
        super();
        this.activity = activity;
        this.intValues = new int[0];
        this.outFrame = new int[0];
        this.tfliteOptions = new Interpreter.Options();
        this.model = new Model("road_segmentation", 225, 225, 225, 225);
        loadModel();
        Log.d(TAG, "Created a Tensorflow Lite Image Segmenter.");

    }

    public Model getModel() {
        return model;
    }

    private void loadModel() throws IOException {
        this.tfliteModel = this.loadModelFile(this.activity);
        this.recreateInterpreter();

    }

    private void recreateInterpreter(){
        Model model = this.model;
        Interpreter tfliteInterperter = this.tflite;
        if(tfliteInterperter != null){
            tfliteInterperter.close();
        }
        MappedByteBuffer tfliteModelin = this.tfliteModel;

        if(tfliteModelin != null){
            this.tflite = new Interpreter((ByteBuffer) tfliteModelin, this.tfliteOptions);
        }
        this.imgData = ByteBuffer.allocateDirect(
                model.getInputWidth()*model.getInputHeight()*DIM_BATCH_SIZE*DIM_PIXEL_SIZE*4)
                .order(ByteOrder.nativeOrder());

        this.segmentedImage = ByteBuffer.allocateDirect(DIM_BATCH_SIZE*model.outputWidth
                *model.outputHeight*4)
                .order(ByteOrder.nativeOrder());

        this.outFrame = new int[model.getOutputWidth() * model.getOutputHeight()];
        this.intValues = new int[model.getInputWidth() * model.getInputHeight()];
    }

    public final void setnNumThreads(int numThreads){
        this.tfliteOptions.setNumThreads(numThreads);
        this.recreateInterpreter();
    }



    private MappedByteBuffer loadModelFile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(this.getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer ML_model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        Log.i("LoadMODELING", "Successfully loaded the model");
        return ML_model;
    }

    private String getModelPath(){
        return this.model.getPath() + ".tflite";
    }

    public void close(){
        if(this.tflite != null){
            this.tflite.close();
        }
        this.tflite = null;
        if(this.tfliteModel != null){
            this.tfliteModel.clear();
        }
        this.tfliteModel = null;
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap){
        this.imgData.rewind();
        bitmap.getPixels(this.intValues, 0,  bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for(int size = this.intValues.length; pixel < size; ++pixel){
            int value = this.intValues[pixel];
            this.imgData.putFloat((float) (value >> 16 & 255));
            this.imgData.putFloat((float) (value >> 8 & 255));
            this.imgData.putFloat((float) (value & 255));
        }
    }

    public int[] segmentFrame(Bitmap bitmap){
        if (this.tflite == null) {
            Log.e("ImageSegmenter", "Image segmenter has not been initialized; Skipped.");
        }
        this.convertBitmapToByteBuffer(bitmap);
        this.segmentedImage.rewind();
        if(this.tflite != null){
            this.tflite.run(this.imgData, this.segmentedImage);
        }
        this.segmentedImage.position(0);
        int i  =0;
        while(this.segmentedImage.hasRemaining()){
            outFrame[i++] = this.segmentedImage.getInt();
        }

        return outFrame;
    }

    //Model Class
    public static class Model{
        private String path;
        private int inputWidth;
        private int inputHeight;
        private int outputWidth;
        private int outputHeight;


        private final Integer[] colors;


        public Model(String path, int inputWidth, int inputHeight, int outputWidth, int outputHeight){
            this.path = path;
            this.colors = new Integer[]{

                    Color.rgb(128, 64, 128) ,
                    Color.rgb(244, 35, 232) ,
                    Color.rgb(70, 70, 70) ,
                    Color.rgb(102, 102, 156) ,
                    Color.rgb(190, 153, 153) ,
                    Color.rgb(153, 153, 153) ,
                    Color.rgb(250, 170, 30) ,
                    Color.rgb(220, 220, 0) ,
                    Color.rgb(107, 142, 35) ,
                    Color.rgb(152, 251, 152) ,
                    Color.rgb(70, 130, 180) ,
                    Color.rgb(220, 20, 60) ,
                    Color.rgb(255, 0, 0) ,
                    Color.rgb(0, 0, 142) ,
                    Color.rgb(0, 0, 70) ,
                    Color.rgb(0, 60, 100) ,
                    Color.rgb(0, 80, 100) ,
                    Color.rgb(0, 0, 230) ,
                    Color.rgb(119, 11, 32)
            };
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;
        };




        public  Integer[] getCityscapesColors() {
            return colors;
        }
        public  String getPath(){
            return  this.path;
        }

        public int getInputHeight() {
            return inputHeight;
        }

        public int getInputWidth() {
            return inputWidth;
        }

        public int getOutputHeight() {
            return outputHeight;
        }

        public int getOutputWidth() {
            return outputWidth;
        }

    }

}

