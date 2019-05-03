package metier.data;

import java.util.Date;
import javax.annotation.Generated;
import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;
import metier.data.Client;
import metier.data.Employe;
import metier.data.Medium;

@Generated(value="EclipseLink-2.5.2.v20140319-rNA", date="2019-05-03T14:50:41")
@StaticMetamodel(Voyance.class)
public class Voyance_ { 

    public static volatile SingularAttribute<Voyance, Date> dateDebut;
    public static volatile SingularAttribute<Voyance, Employe> employe;
    public static volatile SingularAttribute<Voyance, Client> client;
    public static volatile SingularAttribute<Voyance, Integer> id;
    public static volatile SingularAttribute<Voyance, Date> dateFin;
    public static volatile SingularAttribute<Voyance, Medium> medium;
    public static volatile SingularAttribute<Voyance, String> commentaire;
    public static volatile SingularAttribute<Voyance, Date> demandeVoyance;

}