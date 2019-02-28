package com.example.drumpad;

import android.media.MediaPlayer;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final MediaPlayer mediaPlayer = MediaPlayer.create(this, R.raw.clapslapper);
        final MediaPlayer mediaPlayer2 = MediaPlayer.create(this, R.raw.crash808);

        final Button btn = (Button) this.findViewById(R.id.buttonTest);
        final Button btn2 = (Button) this.findViewById(R.id.buttonTest2);
        final Toast t = Toast.makeText(this,"ma queue",Toast.LENGTH_SHORT);

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                t.show();
                mediaPlayer.start();
            }
        });

        btn2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mediaPlayer2.start();
            }
        });

    }
}
