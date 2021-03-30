package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Paint;

import java.lang.reflect.Array;

public class Utils {

    public static void segmentResultToBitmap(int[] segmentedImage, Integer[] classColors, Bitmap targetBitmap){
        int size = targetBitmap.getWidth() * targetBitmap.getWidth();
        int[] pixels = new int[size];
        for (int i = 0; i < size ;i++){
            pixels[i] = classColors[segmentedImage[i]];
        }
        targetBitmap.setPixels(pixels,0, targetBitmap.getWidth(), 0, 0, targetBitmap.getWidth(), targetBitmap.getHeight());

    }
    public static Bitmap resizeBitmap(Bitmap bitmap, int width, int height) throws Throwable {
        Bitmap result = Bitmap.createBitmap(width,height,Config.ARGB_8888);
        Bitmap scaledBitmap;
        if(bitmap.getWidth() != bitmap.getHeight()){
            throw (Throwable)(new Error("Mask expected to be square but got something else"));
        }else{
            if(height > width){
                scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, width, true);
            } else{
                scaledBitmap = Bitmap.createScaledBitmap(bitmap,height, height, true);
            }
            int pX = (width - scaledBitmap.getWidth()) / 2;
            int pY = (height - scaledBitmap.getHeight()) / 2;
            Canvas can = new Canvas(result);
            can.drawBitmap(scaledBitmap, (float) pX, (float) pY, (Paint)null);
            return result;
        }

    }
}

