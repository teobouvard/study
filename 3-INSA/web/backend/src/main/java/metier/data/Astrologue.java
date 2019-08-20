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
public class Astrologue extends Medium{
    private String promotion;
    private String formation;

    public Astrologue() {
    }

    public Astrologue(String nom, String descriptif , String promotion, String formation) {
        super(nom, descriptif);
        this.promotion = promotion;
        this.formation = formation;
    }

    public String getPromotion() {
        return promotion;
    }

    public String getFormation() {
        return formation;
    }
    
    
    
}
