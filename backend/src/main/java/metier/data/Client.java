/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import javax.persistence.*;
import util.AstroTest;
import static util.AstroTest.JSON_DATE_FORMAT;

/**
 *
 * @author ghermet
 */
@Entity
public class Client extends Personne{
    private String signeZodiaque;
    private String signeChinois;
    private String couleur;
    private String animal;
    private Boolean demandeFaite;
    @OneToMany(mappedBy= "client")
    private List<Voyance> listVoyance = new ArrayList<Voyance>();

    public Client() {
    }

    public Client(String nom, String prenom) throws IOException, ParseException {
        super(nom, prenom);
        demandeFaite = false;
        genererProfil();  
    }

    public Client(String civilite, String nom, String prenom, Date date, String adressePostale, String adresseMail, String numero, String password) throws IOException, ParseException {
        super(civilite, nom, prenom, date, adressePostale, adresseMail, numero, password, "Client");
        demandeFaite = false;
        genererProfil();
    }
    
    
    public final void genererProfil () throws IOException, ParseException{
        
        AstroTest astroApi = new AstroTest();
        //Penser à remettre l'attribut date normal
        //Date dateNaiss = JSON_DATE_FORMAT.parse("1976-07-10");
        List<String> profilAstro = astroApi.getProfil(prenom, dateNaissance);
        signeZodiaque = profilAstro.get(0);
        signeChinois = profilAstro.get(1);
        couleur = profilAstro.get(2);
        animal = profilAstro.get(3);
    }

    public void setDemandeFaite(Boolean demandeFaite) {
        this.demandeFaite = demandeFaite;
    }
 
    
    public String getSigneZodiaque() {
        return signeZodiaque;
    }

    public String getSigneChinois() {
        return signeChinois;
    }

    public String getCouleur() {
        return couleur;
    }

    public String getAnimal() {
        return animal;
    }

    public Boolean getDemandeFaite() {
        return demandeFaite;
    }

    public List<Voyance> getListVoyance() {
        return listVoyance;
    }

    public Integer getId() {
        return id;
    }

    public String getCivilite() {
        return civilite;
    }

    public String getNom() {
        return nom;
    }

    public String getPrenom() {
        return prenom;
    }

    public Date getDateNaissance() {
        return dateNaissance;
    }

    public String getAdressePostale() {
        return adressePostale;
    }

    public String getAdresseMail() {
        return adresseMail;
    }

    public String getNumero() {
        return numero;
    }

    public String getPassword() {
        return password;
    }

    public String getType() {
        return type;
    }

    public void ajouterVoyance(Voyance v){
        listVoyance.add(v);
    }
   
   
}
