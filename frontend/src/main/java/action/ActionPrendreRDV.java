package action;

import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Client;
import metier.data.Medium;
import metier.data.Voyance;
import static metier.service.Service.demanderVoyance;
import static metier.service.Service.getListMedium;

public class ActionPrendreRDV extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
        HttpSession session = request.getSession();
        Client client = (Client) session.getAttribute("personne");
        Integer ID = Integer.parseInt(request.getParameter("medium"));
        
        List<Medium> listeMedium = getListMedium();
        Medium medium = null;
        for (Medium m : listeMedium) {
            if (m.getId() == ID) {
                medium = m;
            }
        }

        System.out.println("Nom du client : " + client.getNom());
        System.out.println("Nom du medium : " + medium.getNom());

        Voyance voyance = demanderVoyance(medium, client);
        
        if (voyance != null) {
            System.out.println("Voyance : " + voyance.getEmploye());
            request.setAttribute("statut", Boolean.TRUE);
        } else {
            System.out.println("Voyance pas dispo ? ");
            request.setAttribute("statut", Boolean.FALSE);
        }
        return true;
    }

}
