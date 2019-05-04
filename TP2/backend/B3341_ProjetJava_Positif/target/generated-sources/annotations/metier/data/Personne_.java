package metier.data;

import java.util.Date;
import javax.annotation.Generated;
import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;

@Generated(value="EclipseLink-2.5.2.v20140319-rNA", date="2019-05-03T22:20:33")
@StaticMetamodel(Personne.class)
public class Personne_ { 

    public static volatile SingularAttribute<Personne, String> password;
    public static volatile SingularAttribute<Personne, String> adresseMail;
    public static volatile SingularAttribute<Personne, String> numero;
    public static volatile SingularAttribute<Personne, Date> dateNaissance;
    public static volatile SingularAttribute<Personne, Integer> id;
    public static volatile SingularAttribute<Personne, String> adressePostale;
    public static volatile SingularAttribute<Personne, String> type;
    public static volatile SingularAttribute<Personne, String> nom;
    public static volatile SingularAttribute<Personne, String> prenom;
    public static volatile SingularAttribute<Personne, String> civilite;

}