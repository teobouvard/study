package metier.data;

import javax.annotation.Generated;
import javax.persistence.metamodel.ListAttribute;
import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;
import metier.data.Voyance;

@Generated(value="EclipseLink-2.5.2.v20140319-rNA", date="2019-05-03T22:20:33")
@StaticMetamodel(Client.class)
public class Client_ extends Personne_ {

    public static volatile SingularAttribute<Client, String> signeZodiaque;
    public static volatile SingularAttribute<Client, Boolean> demandeFaite;
    public static volatile ListAttribute<Client, Voyance> listVoyance;
    public static volatile SingularAttribute<Client, String> couleur;
    public static volatile SingularAttribute<Client, String> animal;
    public static volatile SingularAttribute<Client, String> signeChinois;

}