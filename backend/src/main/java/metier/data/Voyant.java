/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;

import javax.persistence.Entity;

/**
 *
 * @author ghermet
 */
@Entity
public class Voyant extends Medium{
    private String specialite;

    public Voyant() {
    }

    public Voyant(String nom, String specialite,  String descriptif) {
        super(nom, descriptif);
        this.specialite = specialite;
    }

    public String getSpecialite() {
        return specialite;
    }

    
}
