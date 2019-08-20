/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.data;

import java.io.Serializable;
import java.util.Date;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.ManyToOne;
import javax.persistence.Temporal;

/**
 *
 * @author ghermet
 */
@Entity
public class Voyance implements Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id;
    @Temporal(javax.persistence.TemporalType.DATE)
    private Date dateDebut;
    @Temporal(javax.persistence.TemporalType.DATE)
    private Date dateFin;
    private String commentaire;
    @Temporal(javax.persistence.TemporalType.DATE)
    private Date demandeVoyance;
    @ManyToOne
    private Employe employe;
    @ManyToOne
    private Client client;
    @ManyToOne
    private Medium medium;

    public Voyance() {
    }

    public Voyance(Date demandeVoyance, Client client, Medium medium) {
        this.demandeVoyance = demandeVoyance;
        this.client = client;
        this.medium = medium;
    }

    public Date getDemandeVoyance() {
        return demandeVoyance;
    }

    public Employe getEmploye() {
        return employe;
    }

    public Client getClient() {
        return client;
    }

    public Medium getMedium() {
        return medium;
    }

    public void setDateDebut(Date dateDebut) {
        this.dateDebut = dateDebut;
    }

    public void setDateFin(Date dateFin) {
        this.dateFin = dateFin;
    }

    public void setCommentaire(String commentaire) {
        this.commentaire = commentaire;
    }

    public void setEmploye(Employe employe) {
        this.employe = employe;
    }

   
    

    public Integer getId() {
        return id;
    }

    public Date getDateDebut() {
        return dateDebut;
    }

    public Date getDateFin() {
        return dateFin;
    }

    public String getCommentaire() {
        return commentaire;
    }

 
    
    
    
    
        
}
