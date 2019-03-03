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


    public ChangerConfig(String instrument){

    }



    public ChangerBoutton(String son, String id){
        configChangee = true;
    }

    public SauverConfig(String name){
        
    }
}
