package com.example.drumpad;

import android.app.Activity;
import android.content.Context;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.felhr.usbserial.UsbSerialDevice;
import com.felhr.usbserial.UsbSerialInterface;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;


public class MainActivity extends Activity {

    private SoundPool soundPool;

    // Maximum sound stream
    private static final int MAX_STREAMS = 16;

    private int soundId0, soundId1, soundId2, soundId3;

    Button startButton, stopButton;
    TextView textView;
    UsbManager usbManager;
    UsbDevice device;
    UsbSerialDevice serialPort;
    UsbDeviceConnection connection;

    UsbSerialInterface.UsbReadCallback mCallback = new UsbSerialInterface.UsbReadCallback() { //Defining a Callback which triggers whenever data is read.
        @Override
        public void onReceivedData(byte[] arg0) {
            String data;
            String padNo = "";
            String power = "";
            try {
                data = new String(arg0, "UTF-8");
                if (data.length() > 0) {
                    padNo = data.substring(0, 1);
                    power = data.substring(1);
                }

                tvAppend(textView, padNo);
                tvAppend(textView, power);
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        usbManager = (UsbManager) getSystemService(Context.USB_SERVICE);
        startButton = (Button) findViewById(R.id.startButton);
        stopButton = (Button) findViewById(R.id.stopButton);
        textView = (TextView) findViewById(R.id.textView);

        soundPool = new SoundPool(MAX_STREAMS, AudioManager.STREAM_MUSIC, 0);
        soundId0 = soundPool.load(this, R.raw.clap0, 1);

    }


    public void onClickStart(View view) {

        HashMap<String, UsbDevice> usbDevices = usbManager.getDeviceList();
        if (!usbDevices.isEmpty()) {
            boolean keep = true;
            for (Map.Entry<String, UsbDevice> entry : usbDevices.entrySet()) {
                device = entry.getValue();
                if (device.getVendorId() == 0x2341)//Arduino Vendor ID
                {
                    connection = usbManager.openDevice(device);
                    serialPort = UsbSerialDevice.createUsbSerialDevice(device, connection);
                    if (serialPort != null) {
                        if (serialPort.open()) { //Set Serial Connection Parameters.
                            serialPort.setBaudRate(9600);
                            //serialPort.setDataBits(UsbSerialInterface.DATA_BITS_8);
                            //serialPort.setStopBits(UsbSerialInterface.STOP_BITS_1);
                            //serialPort.setParity(UsbSerialInterface.PARITY_NONE);
                            //serialPort.setFlowControl(UsbSerialInterface.FLOW_CONTROL_OFF);
                            serialPort.read(mCallback);
                            Toast.makeText(getApplicationContext(), "Serial Connection Opened",
                                    Toast.LENGTH_SHORT).show();
                        }
                        keep = false;
                    } else {
                        connection = null;
                        device = null;
                    }
                    if (!keep)
                        break;
                }
            }
        }
    }

    public void onClickStop(View view) {
        serialPort.close();
        Toast.makeText(getApplicationContext(), "Serial Connection Closed",
                Toast.LENGTH_SHORT).show();
    }

    private void tvAppend(TextView tv, CharSequence text) {
        final TextView ftv = tv;
        final CharSequence ftext = text;

        runOnUiThread(() -> ftv.append(ftext));
    }


    public void button0Click(View view){
        this.soundPool.play(this.soundId0,1, 1, 1, 0, 1f);
    }

/*
    public void button1Click(View view){
        this.soundPool.play(this.soundId1,1, 1, 1, 0, 1f);
    }


    public void button2Click(View view){
        this.soundPool.play(this.soundId2,1, 1, 1, 0, 1f);    }

    public void button3Click(View view){
        this.soundPool.play(this.soundId3,1, 1, 1, 0, 1f);    }

    public void button4Click(View view){
        final MediaPlayer mp4 = MediaPlayer.create(this, R.raw.hihatacoustic01);
        mp4.start();
    }

    public void button5Click(View view){
        final MediaPlayer mp5 = MediaPlayer.create(this, R.raw.kickacoustic02);
        mp5.start();
    }

    public void button6Click(View view){
        final MediaPlayer mp6 = MediaPlayer.create(this, R.raw.kickdeep);
        mp6.start();
    }

    public void button7Click(View view){
        final MediaPlayer mp7 = MediaPlayer.create(this, R.raw.kickvinyl01);
        mp7.start();
    }

    public void button8Click(View view){
        final MediaPlayer mp8 = MediaPlayer.create(this, R.raw.openhat808);
        mp8.start();
    }

    public void button9Click(View view){
        final MediaPlayer mp9 = MediaPlayer.create(this, R.raw.snareacoustic01);
        mp9.start();
    }

    public void button10Click(View view){
        final MediaPlayer mp10 = MediaPlayer.create(this, R.raw.rideacoustic02);
        mp10.start();
    }

    public void button11Click(View view){
        final MediaPlayer mp11 = MediaPlayer.create(this, R.raw.crashacoustic);
        mp11.start();
    }

    public void button12Click(View view){
        final MediaPlayer mp12 = MediaPlayer.create(this, R.raw.clap0);
        mp12.start();
    }

    public void button13Click(View view){
        final MediaPlayer mp13 = MediaPlayer.create(this, R.raw.clap0);
        mp13.start();
    }

    public void button14Click(View view){
        final MediaPlayer mp14 = MediaPlayer.create(this, R.raw.clap0);
        mp14.start();
    }

    public void button15Click(View view){
        final MediaPlayer mp15 = MediaPlayer.create(this, R.raw.clap0);
        mp15.start();
    }
    */
}
