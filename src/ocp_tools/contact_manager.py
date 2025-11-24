import pinocchio as pin
import crocoddyl
import numpy as np

class ContactModelManager:
    """Builds and manages contacts for OCP problem formulation.

    Attributes:
        state (crocoddyl.StateMultibody): The state model of the robot.
        actuation (crocoddyl.ActuationModelAbstract): The actuation model of the robot.
        contact_model_sum (crocoddyl.ContactModelMultiple): The container that holds
            all the individual contact models added to this manager.
    """
    
    def __init__(self, 
                 state: crocoddyl.StateMultibody, 
                 actuation: crocoddyl.ActuationModelAbstract,
                 rmodel: pin.Model,
                 rdata: pin.Data):
        """Initializes the ContactModelManager.

        Args:
            state (crocoddyl.StateMultibody): The state model of the multibody system.
            actuation (crocoddyl.ActuationModelAbstract): The actuation model, used to
                retrieve the control dimension `nu`.
        """
        self.state = state
        self.actuation = actuation
        self.rmodel = rmodel
        self.rdata = rdata

        self.contact_model_sum = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)

    def add_contact_6D(self,
                       frame_name: str,
                       ref_type: pin.ReferenceFrame = pin.LOCAL_WORLD_ALIGNED) -> 'ContactModelManager':
        """Adds a 6D (surface) contact model to the collection.

        This type of contact constrains both the linear and angular velocity
        of a specific frame to zero, effectively modeling a rigid surface contact
        (e.g., a foot flat on the ground).

        Args:
            contact_name (str): A unique name to identify the contact (e.g., "left_foot_contact").
            frame_id (int): The ID of the Pinocchio frame that is in contact.
            ref_placement (pin.SE3): The reference pose (position and orientation) of the
                contact frame in the world frame.

        Returns:
            ContactModelManager: The instance of the manager itself, to allow for method chaining.
        """
        try:
            frame_id = self.rmodel.getFrameId(frame_name)
        except RuntimeError:
            print(f"Erreur : La frame '{frame_name}' n'a pas été trouvée dans le modèle du robot.")
            raise
        
        ref_placement = self.rdata.oMf[frame_id].copy()
        contact_name = f"{frame_name}_contact"
        
        contact_model = crocoddyl.ContactModel6D(self.state,
                                                 frame_id,
                                                 ref_placement,
                                                 ref_type,
                                                 self.actuation.nu,
                                                 np.array([200.0, 20.0]))
        self.contact_model_sum.addContact(contact_name, contact_model)
        
        return self
    
    def display_contacts(self):
        """
        Prints a detailed, human-readable summary of all contacts configured
        in this model.
        """
        print("---- Contact Model Summary ----")
        
        contacts_map = self.contact_model_sum.contacts
        
        if not contacts_map:
            print("  No contacts have been configured in this model.")
            print("---------------------------\n")
            return
            
        print(f"  Total number of contacts: {len(contacts_map)}")
        print("---------------------------")

        for item in contacts_map:
            name = item.key
            contact_item = item.data()

            contact_type = type(contact_item.contact).__name__
            frame_id = contact_item.contact.id
            frame_name = self.rmodel.frames[frame_id].name
            
            print(f"  > Contact Name: '{name}'")
            print(f"    - Type      : {contact_type}")
            print(f"    - Frame Name: '{frame_name}' (ID: {frame_id})")
        
        print("---------------------------\n")
