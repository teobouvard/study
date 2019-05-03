/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Inheritance;
import javax.persistence.InheritanceType;
import javax.persistence.ManyToMany;
import javax.persistence.OneToMany;
/**
 *
 * @author ghermet
 */
@Entity
@Inheritance (strategy = InheritanceType.JOINED)
public abstract class Medium implements Serializable {
    @Id
     @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id; 
    private String nom;
    private String descriptif;
    @ManyToMany
    private List<Employe> listEmploye= new ArrayList<Employe>();
    @OneToMany(mappedBy= "medium")
    private List<Voyance> listVoyance= new ArrayList<Voyance>();

    public Medium() {
    }

    public Medium(String nom, String descriptif) {
        this.nom = nom;
        this.descriptif = descriptif;
    }

    public Integer getId() {
        return id;
    }

    public String getNom() {
        return nom;
    }

    public String getDescriptif() {
        return descriptif;
    }

    public void ajouterEmploye(Employe e){
        listEmploye.add(e);
    }
    public void ajouterVoyance(Voyance v){
        listVoyance.add(v);
    }
    
}


