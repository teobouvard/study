package metier.data;

import javax.annotation.Generated;
import javax.persistence.metamodel.ListAttribute;
import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;
import metier.data.Medium;
import metier.data.Voyance;

@Generated(value="EclipseLink-2.5.2.v20140319-rNA", date="2019-05-03T14:50:41")
@StaticMetamodel(Employe.class)
public class Employe_ extends Personne_ {

    public static volatile ListAttribute<Employe, Voyance> listVoyance;
    public static volatile ListAttribute<Employe, Medium> listMedium;
    public static volatile SingularAttribute<Employe, Boolean> disponible;

}