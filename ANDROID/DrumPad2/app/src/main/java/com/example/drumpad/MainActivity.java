package com.example.drumpad;

import android.annotation.SuppressLint;
import android.media.MediaPlayer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.io.IOException;

import static android.view.MotionEvent.ACTION_DOWN;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setupButtons();
    }

    @SuppressLint("ClickableViewAccessibility")
    private void setupButtons() {
        final Boutton boutton = new Boutton(this,"buttonTest","clapslapper",1);

        boutton.getButton().setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if(event.getActionMasked()==ACTION_DOWN){
                    boutton.stopPlaying();
                    boutton.setPlayerSound();
                    boutton.getMediaPlayer().start();
                }
                return false;
            }
        });
    }

}
