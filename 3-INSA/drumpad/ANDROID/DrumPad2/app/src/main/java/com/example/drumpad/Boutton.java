package com.example.drumpad;


import android.app.Activity;
import android.media.MediaPlayer;
import android.widget.Button;



public class Boutton {
    private Activity activity;
    private Button btn;
    private String id;
    private String mp3;
    private Integer coefVolume;
    private MediaPlayer mediaPlayer;


    public Boutton(Activity activity, String id, String mp3, Integer coefVolume) {
        this.activity = activity;
        this.id = id;
        this.mp3 = mp3;
        this.coefVolume = coefVolume;
        int ressourceId = 0;
        switch(id){
            case "buttonTest":
                ressourceId = R.id.buttonTest;
                break;
        }
        this.btn = activity.findViewById(ressourceId);
        setPlayerSound();
    }

    public Button getButton() {
        return btn;
    }

    public MediaPlayer getMediaPlayer() {
        return mediaPlayer;
    }

    public void setPlayerSound(){
        int soundId = 0;
        switch(mp3){
            case "clapslapper":
                soundId = R.raw.clapslapper;
                break;
        }
        this.mediaPlayer = MediaPlayer.create(activity,soundId);
    }

    public String getId() {
        return id;
    }

    public String getMp3() {
        return mp3;
    }

    public Integer getCoefVolume() {
        return coefVolume;
    }

    public void setCoefVolume(Integer coefVolume) {
        this.coefVolume = coefVolume;
    }

    public void setMediaPlayer(MediaPlayer mediaPlayer) {
        this.mediaPlayer = mediaPlayer;
    }

    public void stopPlaying() {
        if (mediaPlayer != null) {
            mediaPlayer.reset();
            mediaPlayer.stop();
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }


}

