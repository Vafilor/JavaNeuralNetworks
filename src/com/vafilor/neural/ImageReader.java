package com.vafilor.neural;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Andrey Melnikov on 5/14/2016.
 *
 * Reads the test training images and converts them into input vectors.
 */
public class ImageReader
{
    public static List<Matrix> convertData(Path path)
    {
        InputStream input = null;

        try {

            byte[] number = new byte[4];

            input = Files.newInputStream(path);

            input.read(number);

            int magicNumber = getInteger(number);

            input.read(number);

            int images = getInteger(number);

            List<Matrix> imagesAsVectors = new ArrayList<>(images);

            input.read(number);

            int rows = getInteger(number);

            input.read(number);

            int columns = getInteger(number);

            Matrix image = null;

            byte[] pixelValues = new byte[rows * columns * images];

            input.read(pixelValues);

            int pixelsInImage = rows * columns;

            for(int i = 0; i < pixelValues.length; i += pixelsInImage)
            {
                image = new Matrix(rows * columns, 1);

                for (int j = 0; j < pixelsInImage; j++) {
					//We bitwise an here to convert java's signed byte to 'unsigned' byte, really an int here.
					//Also we divide by 256.0 to make the value be from [0,1), as the network should have each output in that range.
                    image.setEntry(j, 0, (pixelValues[i + j] & 0xFF) / 256.0); 
                }

                imagesAsVectors.add(image);
            }

            return imagesAsVectors;

        } catch(IOException exception)
        {
            System.out.format("Error in reading file:" + exception.getMessage());
            return new ArrayList<>();
        } finally
        {
            if( null != input )
            {
                try {
                    input.close();
                }
                catch (IOException exception)
                {
                    System.err.format("Failed to close input stream for %s. Error %s", path, exception.getMessage());
                }
            }
        }
    }

    private static int getInteger(byte[] number)
    {
        return ByteBuffer.wrap(number).getInt();
    }
}
