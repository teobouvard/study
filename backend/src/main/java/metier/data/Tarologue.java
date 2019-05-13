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
public class Tarologue extends Medium {

    public Tarologue() {
    }

    public Tarologue(String nom, String descriptif) {
        super(nom, descriptif);
    }
    
    
    
}
