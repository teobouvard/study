/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;

import java.io.Serializable;
import java.util.Date;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Inheritance;
import javax.persistence.InheritanceType;
import javax.persistence.Temporal;

/**
 *
 * @author ghermet
 */
@Entity
@Inheritance (strategy = InheritanceType.JOINED)
public class Personne implements Serializable {


    @Id
     @GeneratedValue(strategy = GenerationType.AUTO)
    protected Integer id;
    protected String civilite;
    protected String nom;
    protected String prenom;
    @Temporal(javax.persistence.TemporalType.DATE)
    protected Date dateNaissance;
    protected String adressePostale;
    @Column(unique = true)
    protected String adresseMail;
    protected String numero;
    protected String password;
    protected String type;

    public Personne() {
    }

    public Personne(String civilite, String nom, String prenom, Date date, String adressePostale, String adresseMail, String numero, String password, String type) {
        this.civilite = civilite;
        this.nom = nom;
        this.prenom = prenom;
        this.dateNaissance = date;
        this.adressePostale = adressePostale;
        this.adresseMail = adresseMail;
        this.numero = numero;
        this.password = password;
        this.type = type;
    }

    public Personne(String nom, String prenom) {
        this.nom = nom;
        this.prenom = prenom;
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

}
