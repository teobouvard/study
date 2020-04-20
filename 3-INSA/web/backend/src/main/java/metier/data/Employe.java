/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import javax.persistence.Entity;
import javax.persistence.ManyToMany;
import javax.persistence.OneToMany;

/**
 *
 * @author ghermet
 */
@Entity
public class Employe extends Personne{
    
    private Boolean disponible;
    @ManyToMany
    private List<Medium> listMedium = new ArrayList<Medium>();
    @OneToMany(mappedBy= "employe")
    private List<Voyance> listVoyance = new ArrayList<Voyance>();

    public Employe() {
    }

    public Employe(String nom, String prenom) {
        super(nom, prenom);
        this.disponible = true;
    }

    public Employe(String civilite, String nom, String prenom, Date date, String adressePostale, String adresseMail, String numero, String password) {
        super(civilite, nom, prenom, date, adressePostale, adresseMail, numero, password, "Employe");
        disponible = true;
    }
    
    

    public List<Voyance> getListVoyance() {
        return listVoyance;
    }
    
    
    
    public Boolean getDisponible() {
        return disponible;
    }

    public void setDisponible(Boolean disponible) {
        this.disponible = disponible;
    }
    
    

    
    public void ajouterVoyance(Voyance v){
        listVoyance.add(v);
    }

    public void ajouterMedium(Medium m){
        listMedium.add(m);
    }
    
    public boolean trouverMedium(Medium med){
        for (Medium element: listMedium){
        
            if(element.getId()== med.getId()){
                return true;
            }  
        }
        return false;  
    }
}

