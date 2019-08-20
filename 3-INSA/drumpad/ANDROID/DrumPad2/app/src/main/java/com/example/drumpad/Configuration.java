package com.example.drumpad;

import java.util.ArrayList;

public class Configuration {

    private ArrayList<Boutton> listeBouttons;
    private String instrumentActuel;
    private boolean configChangee;

    public Configuration(){
        //charger l'instrument par défaut

        configChangee = false;
    }


    public void ChangerConfig(String instrument){

    }



    public void ChangerBoutton(String son, String id){
        configChangee = true;
    }

    public void SauverConfig(String name){

    }
}
