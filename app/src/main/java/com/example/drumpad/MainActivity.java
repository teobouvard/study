package com.example.drumpad;

import android.media.MediaPlayer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    MediaPlayer mp0,mp1,mp2,mp3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mp0 = MediaPlayer.create(this, R.raw.clap0);
        mp1 = MediaPlayer.create(this, R.raw.clapslapper);
        mp2 = MediaPlayer.create(this, R.raw.crash808);
        mp3 = MediaPlayer.create(this, R.raw.crashnoise);
    }


    public void button0Click(View view){
        mp0.start();
    }

    public void button1Click(View view){
        mp1.start();
    }

    public void button2Click(View view){
        mp2.start();
    }

    public void button3Click(View view){
        mp3.start();
    }

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
}
