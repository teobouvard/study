package action;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.servlet.http.HttpServletRequest;
import metier.data.Client;
import static metier.service.Service.ajouterClient;

public class ActionInscrire extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {

        String password = request.getParameter("password");
        String civilite = request.getParameter("civilite");
        String nom = request.getParameter("nom");
        String prenom = request.getParameter("prenom");
        String dateNaissance = request.getParameter("dateNaissance");
        String telephone = request.getParameter("telephone");
        String adresse = request.getParameter("poste");
        String mail = request.getParameter("mail");

        try {
            SimpleDateFormat dateParser = new SimpleDateFormat("yyyy-MM-dd");
            Date date = dateParser.parse(dateNaissance);
            Client nouveauClient = new Client(civilite, nom, prenom, date, adresse, mail, telephone, password);
            ajouterClient(nouveauClient);
            request.setAttribute("statut", true);
            return true;
        } catch (ParseException | IOException ex) {
            request.setAttribute("statut", false);
            return false;
        }
    }

}
