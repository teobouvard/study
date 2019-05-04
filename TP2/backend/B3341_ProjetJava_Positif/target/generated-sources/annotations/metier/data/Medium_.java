package metier.data;

import javax.annotation.Generated;
import javax.persistence.metamodel.ListAttribute;
import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;
import metier.data.Employe;
import metier.data.Voyance;

@Generated(value="EclipseLink-2.5.2.v20140319-rNA", date="2019-05-03T22:20:33")
@StaticMetamodel(Medium.class)
public abstract class Medium_ { 

    public static volatile ListAttribute<Medium, Employe> listEmploye;
    public static volatile ListAttribute<Medium, Voyance> listVoyance;
    public static volatile SingularAttribute<Medium, Integer> id;
    public static volatile SingularAttribute<Medium, String> nom;
    public static volatile SingularAttribute<Medium, String> descriptif;

}