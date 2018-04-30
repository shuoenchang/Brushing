/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.brush.demo;

import android.net.Uri;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.content.SharedPreferences;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.List;
import java.util.Vector;
import org.brush.demo.OverlayView.DrawCallback;
import org.brush.demo.env.BorderedText;
import org.brush.demo.env.ImageUtils;
import org.brush.demo.env.Logger;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  //
  // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
  // model first:
  //
  // python strip_unused.py \
  // --input_graph=<retrained-pb-file> \
  // --output_graph=<your-stripped-pb-file> \
  // --input_node_names="Mul" \
  // --output_node_names="final_result" \
  // --input_binary=truew
  private static final int INPUT_SIZE = 150;
  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 1f;
  private static final String INPUT_NAME = "input_1";
  private static final String OUTPUT_NAME = "dense_2/Softmax";

  private static final String MODEL_FILE = "file:///android_asset/rounded_vgg.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/constant_labels.txt";

  private static final boolean SAVE_PREVIEW_BITMAP = true;

  private static final boolean MAINTAIN_ASPECT = true;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(150, 150);

  private Classifier classifier;

  private Integer sensorOrientation;

  private int count = 0;
  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private Bitmap cropCopyBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private ResultsView resultsView;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private static final float TEXT_SIZE_DIP = 10;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    classifier =
        TensorFlowImageClassifier.create(
            getAssets(),
            MODEL_FILE,
            LABEL_FILE,
            INPUT_SIZE,
            IMAGE_MEAN,
            IMAGE_STD,
            INPUT_NAME,
            OUTPUT_NAME);

    resultsView = (ResultsView) findViewById(R.id.results);
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbBytes = new int[previewWidth * previewHeight];
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            INPUT_SIZE, INPUT_SIZE,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    yuvBytes = new byte[3][];

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            renderDebug(canvas);
          }
        });
  }
  //POST
  public void sentDataPost(final String mData) {
    new Thread(new Runnable() {
      @Override
      public void run() {
        try {

            SharedPreferences prefs = getSharedPreferences("Prefer", Context.MODE_PRIVATE);
            boolean checked = prefs.getBoolean("givenData", true);
            Log.d("check", Boolean.toString(checked));


            if(!checked) {    // Don't agree to collected Data
                Log.d("UploadTarget", "ByeBye");
                return;
            }
            String uri = Uri.parse("http://140.123.97.110:1337/brushTeeth/uploads/imgPost.php")
                  .buildUpon()
                  .build().toString();

            Log.d("UploadTarget", uri);


          URL url = new URL(uri);

          HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
          urlConnection.setRequestMethod("POST");
          urlConnection.setRequestProperty("Connection", "Keep-Alive");
          urlConnection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

          BufferedOutputStream out = new BufferedOutputStream(urlConnection.getOutputStream());

          BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, "UTF-8"));

          writer.write("data=" + URLEncoder.encode(mData, "utf-8"));
          writer.flush();
          writer.close();

          InputStream inputStream = urlConnection.getInputStream();
          BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
          String tempStr;
          StringBuffer stringBuffer = new StringBuffer();

          while ((tempStr = bufferedReader.readLine()) != null) {
            stringBuffer.append(tempStr);
          }

          bufferedReader.close();
          inputStream.close();
          urlConnection.disconnect();
          Log.d("hi", stringBuffer.toString());

        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }).start();
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    try {

      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (computing) {
        image.close();
        return;
      }
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          rgbBytes);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.


    final String filename = "brush" + count;
      count += 1;
    FileOutputStream out = null;
    try {
      out = new FileOutputStream(filename);
      croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
      // PNG is a lossless format, the compression factor (100) is ignored

      //POST upload
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
      byte[] b = baos.toByteArray();
      String encodeImage = Base64.encodeToString(b ,Base64.DEFAULT);
      sentDataPost(encodeImage);

    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      try {
        if (out != null) {
          out.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap,filename+".png");
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
      byte[] b = baos.toByteArray();
      String encodeImage = Base64.encodeToString(b ,Base64.DEFAULT);
      sentDataPost(encodeImage);
    }
    //setContentView(R.layout.camera_connection_fragment);


    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            resultsView.setResults(results);

            /*------------------Teeth model------------------*/
            runOnUiThread(new Runnable() {
              @Override
              public void run() {
                String fileName = "tooth_" + results.get(0).getTitle().charAt(0)
                        + results.get(0).getTitle().charAt(1);
                int resID = getResources().getIdentifier(fileName , "drawable", getPackageName());
                ImageView img = (ImageView) findViewById(R.id.tooth_view);
                img.setImageResource(resID);

              Log.i("TextOutput", fileName);
            }
        });


            requestRender();
            computing = false;
          }
        });

    Trace.endSection();
  }

  @Override
  public void onSetDebug(boolean debug) {
    classifier.enableStatLogging(debug);
  }

  private void renderDebug(final Canvas canvas) {
    if (!isDebug()) {
      return;
    }
    final Bitmap copy = cropCopyBitmap;
    if (copy != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor = 2;
      matrix.postScale(scaleFactor, scaleFactor);
      matrix.postTranslate(
          canvas.getWidth() - copy.getWidth() * scaleFactor,
          canvas.getHeight() - copy.getHeight() * scaleFactor);
      canvas.drawBitmap(copy, matrix, new Paint());

      final Vector<String> lines = new Vector<String>();
      if (classifier != null) {
        String statString = classifier.getStatString();
        String[] statLines = statString.split("\n");
        for (String line : statLines) {
          lines.add(line);
        }
      }

      lines.add("Frame: " + previewWidth + "x" + previewHeight);
      lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
      lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
      lines.add("Rotation: " + sensorOrientation);
      lines.add("Inference time: " + lastProcessingTimeMs + "ms");

      borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
  }
}
